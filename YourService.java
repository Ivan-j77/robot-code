package jp.jaxa.iss.kibo.rpc.defaultapk;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {
    private final String TAG = this.getClass().getSimpleName();

    // ---------------------------------- Config ----------------------------------
    private static final int    PIX_THRESHOLD           = 25;     // pixel offset threshold
    private static final int    CROP_MARGIN             = 50;     // px margin for tight-crop
    private static final double MAX_RAD_ADJ_SPECIAL     = 0.08;   // ≈4.6° per iteration
    private static final int    AUTOAIM_ITER_SPECIAL    = 8;      // up to 8 iterations
    private static final double BACKUP_DIST             = 0.15;   // m to back off when Tag not found
    private static final boolean DEBUG                 = true;
    private static final boolean SAVE_PROCESSED_IMAGE   = true;
    private static final int    IMAGE_SIZE             = 640;    // px for Bitmap and reshape

    // map areaId (1–5) → actual tag id on the wall
    private static final int[] AREA_TO_TAG = {
            0,   // dummy idx-0, never used
            101, // AREA1
            102, // AREA2
            103, // AREA3
            104, // AREA4
            100  // AREA5  (Astronaut)
    };

    // 11 labels
    private final String[] LABEL_MAP = {
            "crystal","diamond","emerald",
            "coin","compass","coral","fossil","key","letter","shell","treasure_box"
    };

    private DetectorNode detectorNode;
    private final PathSequence pathSequence = new PathSequence();
    private String findItem;

    // Landmark whitelist
    private boolean isLandmark(String label) {
        return Arrays.asList(
                "coin","compass","coral","fossil","key","letter","shell","treasure_box"
        ).contains(label);
    }

    // return expected tag id for given area (1-based); 0 → unknown
    private int tagIdForArea(int areaId) {
        if (areaId < 1 || areaId >= AREA_TO_TAG.length) return 0;
        return AREA_TO_TAG[areaId];
    }

    @Override
    protected void runPlan1() {
        api.flashlightControlFront(0.0f);
        api.startMission();
        Log.i(TAG, "Mission started");

        try {
            detectorNode = new DetectorNode(getApplicationContext());
            Log.i(TAG, "DetectorNode initialized");

            movePoints();
            Log.i(TAG, "Initial sweep finished");

            retrievePath();
            takeSnapshotEarly();
            Log.i(TAG, "Mission completed");
        } catch (Exception e) {
            Log.e(TAG, "Fatal error – " + e.getMessage(), e);
            takeSnapshotEarly();
        }
    }

    // =========================================================================
    // Sweep through AREA1–4, record all detected items, report only Landmarks
    // =========================================================================
    private void movePoints() throws InterruptedException {
        Log.i(TAG, "movePoints: start");

        // Camera intrinsics
        Mat K = new Mat(3, 3, CvType.CV_64F);
        K.put(0, 0, api.getNavCamIntrinsics()[0]);
        Mat D = new Mat(1, 5, CvType.CV_64F);
        D.put(0, 0, api.getNavCamIntrinsics()[1]);
        D.convertTo(D, CvType.CV_64F);

        // Init undistort map
        Mat initFrame = api.getMatNavCam();
        initUndistortMap(K, D, initFrame.cols(), initFrame.rows());

        // Four waypoints
        Point[] pts = {
                new Point(10.89995,-9.77284,5.19500),
                new Point(11.00096,-8.82368,4.49984),
                new Point(11.10995,-7.80000,4.48000),
                new Point(10.70933,-6.96802,4.80749)
        };
        Quaternion[] qts = {
                new Quaternion(0f,0f,-0.707f,0.707f),
                new Quaternion(0.017f,0.721f,0f,0.692f),
                new Quaternion(0f,0.707f,0f,0.707f),
                new Quaternion(0.025f,0f,-0.999f,0.04f)
        };

        for (int i = 0; i < 4; i++) {
            int areaId = i + 1;
            //int tarAruco = areaId + 100;
            Log.i(TAG, String.format("movePoints: Moving to AREA%d", areaId));
            api.moveTo(pts[i], qts[i], false);

            // capture raw and undistort
            Mat raw = api.getMatNavCam();
            Mat und = remapUndistort(raw);
            saveMatIfDebug(raw, String.format("p%d_first.png", areaId));
            saveMatIfDebug(und, String.format("p%d_first_und.png", areaId));

            // auto-aim
            Quaternion afterAim = aimToTag(pts[i], qts[i], K, D, areaId);

            // capture after aim
            Mat rawAimed = api.getMatNavCam();
            Mat undAimed = remapUndistort(rawAimed);
            saveMatIfDebug(rawAimed, String.format("p%d_aim.png", areaId));
            saveMatIfDebug(undAimed, String.format("p%d_aim_und.png", areaId));

            // --- DETECT ALL ITEMS ---
            Mat proc = preprocessImage(undAimed, true, areaId);
            Log.i(TAG, "preprocessImageOK, proc: "+ (proc != null));
            saveMatIfDebug(proc, String.format("p%d_processed.png", areaId));
            Bitmap bmp = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(proc, bmp);
            Bitmap drawBmp = bmp.copy(Bitmap.Config.ARGB_8888, true);

            detectorNode.run(bmp, drawBmp, true);
            int[] counts = detectorNode.lastClassCnt;
            Log.i(TAG, String.format("AREA%d detection counts: %s", areaId, Arrays.toString(counts)));
            saveBitmapIfDebug(drawBmp, String.format("p%d_detected.png", areaId));

            for (int idx = 0; idx < counts.length; idx++) {
                int cnt = counts[idx];
                if (cnt <= 0) continue;
                String label = LABEL_MAP[idx];
                // record internally
                pathSequence.recordPath(pts[i], afterAim, label);
                Log.i(TAG, String.format("Recorded [%s] x%d at AREA%d", label, cnt, areaId));
                // only report landmark
                if (isLandmark(label)) {
                    api.setAreaInfo(areaId, label, cnt);
                    Log.i(TAG, String.format("API.setAreaInfo AREA%d %s x%d", areaId, label, cnt));
                }
            }
        }

        // =========================================================================
        // Final Astronaut pose, no auto-aim
        // =========================================================================
        Point pFinal     = Coordinates.ASTRONAUT.getPoint();
        Quaternion qFinal = Coordinates.ASTRONAUT.getQuaternion();
        Log.i(TAG, "movePoints: Move to final (Astronaut) WITHOUT auto-aim");
        api.moveTo(pFinal, qFinal, false);
        api.reportRoundingCompletion();
        api.notifyRecognitionItem();

        // final detection
        int maxRetries       = 3;
        String detected = "NONE";
        for (int k = 1; k <= maxRetries; k++) {
            Mat rawF = api.getMatNavCam();
            Mat undF = remapUndistort(rawF);
            saveMatIfDebug(rawF, String.format("final_raw_%d.png", k));
            saveMatIfDebug(undF, String.format("final_und_%d.png", k));


            Bitmap bmp = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888);
            Mat prcundF = preprocessImage(undF, true, 5);
            saveMatIfDebug(prcundF, "p5_processed.png");
            Utils.matToBitmap(prcundF, bmp);
            Bitmap drawundF = bmp.copy(Bitmap.Config.ARGB_8888, true);

            int[] got = detectorNode.run(bmp, drawundF, true);
            int id = got[1];
            detected = LABEL_MAP[id];
            saveBitmapIfDebug(drawundF, "p5_detected.png");

            Thread.sleep(1000);
        }
        findItem = detected;
        Log.i(TAG, "movePoints: Final found = " + findItem);
        Log.i(TAG, "movePoints: End");
    }

    // =============================================================================

    // ========================= preprocessImage  =========================
    public Mat preprocessImage(Mat src, boolean getBounded, int areaId) {
        // Clone the original image
        Mat working = src.clone();

        // Detect all tags
        Transform.ArTagResult result = Transform.getARTag(src);
        LinkedList<Mat> allCorners = result.corners;
        int[] ids = result.markerIds;

        // Filter out the tag that matches the expected area
        Mat targetCorner = null;
        if (allCorners != null && ids != null) {
            int expectedId = tagIdForArea(areaId);;  // AREA1→101, AREA2→102...
            for (int i = 0; i < ids.length; i++) {
                if (ids[i] == expectedId) {
                    targetCorner = allCorners.get(i);
                    break;
                }
            }
        }

        // Use the matched tag if found; otherwise fallback to the first one
        if (targetCorner != null) {
            allCorners = new LinkedList<>();
            allCorners.add(targetCorner);
        }

        // Continue with the pipeline using the updated corners
        if (allCorners != null && !allCorners.isEmpty()) {
            Mat rot = Transform.rotateToParallel(src.clone(), allCorners.getFirst());

            Transform.ArTagResult result2 = Transform.getARTag(rot);
            LinkedList<Mat> corners2 = pickSameIdCorner(result2, areaId); // Again, filter by ID
            if (corners2 != null && !corners2.isEmpty()) {
                Mat cropped = Transform.cropMarkedImage(rot, corners2);

                if (getBounded) {
                    Transform.ArTagResult result3 = Transform.getARTag(cropped);
                    LinkedList<Mat> corners3 = pickSameIdCorner(result3, areaId);
                    if (corners3 != null && !corners3.isEmpty()) {
                        int[] bound = Transform.getBoundedRegion(cropped, corners3.getFirst());
                        cropped = safeSubmat(cropped, bound, CROP_MARGIN);
                    }
                }
                working = cropped;
            }
        } else {
            Log.i(TAG, "preprocessImage: AREA" + areaId + " - Expected Tag not found");
        }

        // Apply filter → resize → reshape
        Transform.filter2D(working, working);
        working = Transform.resizeImg(working, 640);
        working = Transform.reshapeImage(working, IMAGE_SIZE, IMAGE_SIZE);
        return working;
    }

    // Utility function: retrieve tag corner by marker ID
    private LinkedList<Mat> pickSameIdCorner(Transform.ArTagResult res, int areaId) {
        if (res == null || res.corners == null || res.markerIds == null) return null;
        int expectedId = tagIdForArea(areaId);;
        LinkedList<Mat> list = new LinkedList<>();
        for (int i = 0; i < res.markerIds.length; i++) {
            if (res.markerIds[i] == expectedId) list.add(res.corners.get(i));
        }
        return list;
    }

    // Safe cropping
    private Mat safeSubmat(Mat src, int[] bound, int margin) {
        int r0 = Math.max(0, bound[1] - margin);
        int r1 = Math.min(src.rows(), bound[3] + margin);
        int c0 = Math.max(0, bound[0] - margin);
        int c1 = Math.min(src.cols(), bound[2] + margin);
        return (r0 < r1 && c0 < c1) ? src.submat(r0, r1, c0, c1) : src;
    }


    // =============================================================================

    // ======================= Undistort map init & remap =======================
    private Mat undistMap1, undistMap2; private boolean undistInited=false;
    private void initUndistortMap(Mat K, Mat D, int w, int h) {
        if (undistInited) return;
        undistMap1=new Mat(); undistMap2=new Mat();
        Calib3d.initUndistortRectifyMap(K,D,new Mat(),K,new Size(w,h),
                CvType.CV_32FC1,undistMap1,undistMap2);
        undistInited=true;
        Log.i(TAG, "initUndistortMap: done");
    }

    private Mat remapUndistort(Mat src) {
        Mat dst=new Mat();
        Imgproc.remap(src, dst, undistMap1, undistMap2, Imgproc.INTER_LINEAR);
        return dst;
    }
    // =============================================================================

    /* ====================================================================== */
    /*  AIM TO TAG – one-shot basic + backup special-mode                      */
    /* ====================================================================== */

    /**
     * Try one-shot basic aim; if first frame has no AR-Tag,
     * perform backup special flow: back off BACKUP_DIST → scan → special-aim →
     * move forward BACKUP_DIST.
     */

    // ============================ Auto-aim routines ============================
    private Quaternion aimToTag(Point pose, Quaternion q, Mat K, Mat D, int areaId) {
        Mat first = api.getMatNavCam(), und = remapUndistort(first);
        saveMatIfDebug(first, String.format("p%d_aimToTag_first_raw.png",areaId)); 
        saveMatIfDebug(und, String.format("p%d_aimToTag_first_und.png",areaId));

        Transform.ArTagResult result = Transform.getARTag(und);
        LinkedList<Mat> corners = result.corners;
        int[] IDS = result.markerIds;
        Mat cnr = null;

        if (corners != null)

            for (int i = 0; i < IDS.length; i++){
                if (IDS[i] == areaId + 100){
                    Log.i(TAG, "right aruco");
                    cnr = corners.get(i);
                }
            }

            if (cnr!=null) {
                Log.i(TAG, "aimToTag: basic");
                return autoAimBasicOnce(pose, q, K, D, areaId);
            }

        Log.i(TAG, "aimToTag: fallback special");
        // back off
        Point back = offsetPointBack(pose, q, BACKUP_DIST);
        api.moveTo(back, q, false);
        Log.i(TAG, "Backed off");
        Mat backF = api.getMatNavCam(), undB=remapUndistort(backF);

        saveMatIfDebug(backF,String.format("p%d_aimToTag_backoff_raw.png", areaId));
        saveMatIfDebug(undB,String.format("p%d_aimToTag_backoff_und.png", areaId));

        Transform.ArTagResult resultB = Transform.getARTag(undB);
        LinkedList<Mat> backCorners = resultB.corners;

        if (backCorners==null || backCorners.isEmpty()) {
             Log.i(TAG, "aimToTag: AR-Tag found, running BASIC one-shot aim");
            api.moveTo(pose, q, false); return q;
        }
        Quaternion q2 = autoAimSpecialIterative(back, q, K, D, areaId);
        api.moveTo(pose, q2, false);
        return q2;
    }

    /* ====================================================================== */
    /*  ONE-SHOT BASIC AUTO-AIM (arctan-based)                                 */
    /* ====================================================================== */

    /**
     * One-shot basic auto-aim: grab a frame, compute exact yaw/pitch via atan,
     * move directly if outside PIX_THRESHOLD.
     */

    private Quaternion autoAimBasicOnce(
            Point pose,
            Quaternion startQ,
            Mat K, Mat D, int arucoId) {

        // Grab & undistort a frame
        Mat img = api.getMatNavCam();
        Mat und = remapUndistort(img);
        // saveMatIfDebug(img, "basicOnce_raw.png");
        // saveMatIfDebug(und, "basicOnce_undist.png");

        // Detect AR-Tag corners
        Transform.ArTagResult result = Transform.getARTag(und);
        LinkedList<Mat> corners = result.corners;
        int[] IDS = result.markerIds;
        Mat cnr = null;
        Log.i(TAG, "targetId:" + (arucoId + 100));

        if (corners != null )

            for (int i = 0; i < IDS.length; i++){
                if (IDS[i] == arucoId + 100){
                    cnr = corners.get(i);
                    Log.i(TAG, "id:" + IDS[i] + "correct");
                    break;

                }
            }

        if (cnr == null) {
            Log.i(TAG, "autoAimBasicOnce: AR-Tag not found, skipping basic aim");
            return startQ;
        }

        // Compute pixel center of tag
        double sumX = 0, sumY = 0;
        int cnt = 0;

        for (int j = 0; j < 4; j++) {
            double[] pt = cnr.get(j, 0);
            if (pt != null && pt.length >= 2) {
                sumX += pt[0];
                sumY += pt[1];
                cnt++;
            }
        }
        Log.i(TAG, "detected " + cnt + " corners");
        double u = sumX / cnt;
        double v = sumY / cnt;
        double cx = K.get(0, 2)[0];
        double cy = K.get(1, 2)[0];
        double dx = u - cx;
        double dy = v - cy;


        Log.i(TAG, String.format(
                "autoAimBasicOnce: u,v=(%.1f,%.1f)  dx=%.1f  dy=%.1f", u, v, dx, dy));

        // Compute exact angles via arctan
        double fx = K.get(0, 0)[0];
        double fy = K.get(1, 1)[0];
        double yaw   = Math.atan(dx / fx);    // left +, right –
        double pitch = -Math.atan(dy / fy);   // up +, down –

        // If within PIX_THRESHOLD, skip moving
        if (Math.abs(dx) < PIX_THRESHOLD && Math.abs(dy) < PIX_THRESHOLD) {
            Log.i(TAG, "autoAimBasicOnce: Within PIX_THRESHOLD, no move needed");
            return startQ;
        }

        // Convert to quaternion and move once
        Quaternion qCorr = eulerToQuaternion(0.0, pitch, yaw);
        Quaternion newQ = multiplyQuaternion(startQ, qCorr);
        api.moveTo(pose, newQ, false);
        Log.i(TAG, String.format(
                "autoAimBasicOnce: Moved with yaw=%.3f rad, pitch=%.3f rad", yaw, pitch));

        return newQ;
    }

    /* ====================================================================== */
    /*  SPECIAL ITERATIVE AUTO-AIM (arctan-based)                              */
    /* ====================================================================== */

    /**
     * Robust special-mode auto-aim.
     * 1. Each iteration captures an image → undistorts → detects AR-Tag.
     * 2. Only retains the tag with id == areaId + 100.
     * 3. Calculates yaw/pitch, clamps to ±MAX_RAD_ADJ_SPECIAL, then moves.
     * 4. Ends when pixel error < PIX_THRESHOLD or max iterations reached.
     *
     * Handles the following cases:
     *   • res.corners is null
     *   • Target tag not found
     *   • corner.get() returns null (some corners are missing)
     */

    private Quaternion autoAimSpecialIterative(
            Point pose,
            Quaternion startQ,
            Mat K, Mat D,
            int areaId) {

        final double fx = K.get(0, 0)[0];
        final double fy = K.get(1, 1)[0];
        final double cx = K.get(0, 2)[0];
        final double cy = K.get(1, 2)[0];

        Quaternion currentQ = startQ;
        final int expectedId = tagIdForArea(areaId);;

        for (int iter = 0; iter < AUTOAIM_ITER_SPECIAL; iter++) {

            // Capture and undistort image
            Mat img   = api.getMatNavCam();
            Mat und   = remapUndistort(img);
            saveMatIfDebug(img, String.format("p%d_special_iter_raw.png", areaId));
            saveMatIfDebug(und, String.format("p%d_special_iter_und.png", areaId));

            // Detect ArUco tags
            Transform.ArTagResult res = Transform.getARTag(und);
            if (res == null || res.corners == null || res.markerIds == null
                    || res.corners.isEmpty()) {
                Log.i(TAG, "iter " + iter + " – no tag found, retrying");
                sleepQuiet(300);
                continue;
            }

            // Select the tag with the expected ID
            Mat targetCorner = null;
            for (int i = 0; i < res.markerIds.length; i++) {
                if (res.markerIds[i] == expectedId) {
                    targetCorner = res.corners.get(i);
                    break;
                }
            }
            if (targetCorner == null) {
                Log.i(TAG, "iter " + iter + " – tag ID mismatch, retrying");
                sleepQuiet(300);
                continue;
            }

            // Compute the pixel center safely
            double sumX = 0, sumY = 0;
            int valid = 0;
            for (int j = 0; j < 4; j++) {
                double[] pt = targetCorner.get(j, 0);
                if (pt != null && pt.length >= 2) { // Prevent null arrays
                    sumX += pt[0];
                    sumY += pt[1];
                    valid++;
                }
            }
            if (valid < 2) { // Not enough valid corners, retry
                Log.i(TAG, "iter " + iter + " – corners missing, retrying");
                sleepQuiet(300);
                continue;
            }
            double u  = sumX / valid;
            double v  = sumY / valid;
            double dx = u - cx;
            double dy = v - cy;

            Log.i(TAG, String.format(
                    "iter=%d  u=%.1f v=%.1f  dx=%.1f dy=%.1f", iter, u, v, dx, dy));

            // Check for convergence
            if (Math.abs(dx) < PIX_THRESHOLD && Math.abs(dy) < PIX_THRESHOLD) {
                Log.i(TAG, "iter " + iter + " – alignment OK");
                break;
            }

            // Calculate and clamp the adjustment rotation
            double yaw   =  Math.atan(dx / fx);
            double pitch = -Math.atan(dy / fy);
            yaw   = Math.max(-MAX_RAD_ADJ_SPECIAL, Math.min(MAX_RAD_ADJ_SPECIAL, yaw));
            pitch = Math.max(-MAX_RAD_ADJ_SPECIAL, Math.min(MAX_RAD_ADJ_SPECIAL, pitch));

            Quaternion dq = eulerToQuaternion(0.0, pitch, yaw);
            currentQ = multiplyQuaternion(currentQ, dq);
            api.moveTo(pose, currentQ, false);

            Log.i(TAG, String.format(
                    "iter=%d  moved (yaw=%.3f rad, pitch=%.3f rad)", iter, yaw, pitch));

            sleepQuiet(400);
        }
        return currentQ;
    }

    // Helper to sleep without throwing
    private void sleepQuiet(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignore) {}
    }



    // ============================================================================

    // ====================== Geometry & Quaternion Helpers ======================
    private Point offsetPointBack(Point p, Quaternion q, double dist) {
        double x=q.getX(),y=q.getY(),z=q.getZ(),w=q.getW();
        double fx=1-2*(y*y+z*z), fy=2*(x*y+z*w), fz=2*(x*z-y*w);
        double norm=Math.sqrt(fx*fx+fy*fy+fz*fz); if(norm==0) norm=1;
        fx/=norm; fy/=norm; fz/=norm;
        return new Point(p.getX()-dist*fx, p.getY()-dist*fy, p.getZ()-dist*fz);
    }

    private Quaternion eulerToQuaternion(double roll, double pitch, double yaw) {
        double cr=Math.cos(roll*0.5), sr=Math.sin(roll*0.5);
        double cp=Math.cos(pitch*0.5), sp=Math.sin(pitch*0.5);
        double cy=Math.cos(yaw*0.5), sy=Math.sin(yaw*0.5);
        double w=cr*cp*cy+sr*sp*sy;
        double x=sr*cp*cy-cr*sp*sy;
        double y=cr*sp*cy+sr*cp*sy;
        double z=cr*cp*sy-sr*sp*cy;
        return new Quaternion((float)x,(float)y,(float)z,(float)w);
    }

    private Quaternion multiplyQuaternion(Quaternion a, Quaternion b) {
        float w1=a.getW(),x1=a.getX(),y1=a.getY(),z1=a.getZ();
        float w2=b.getW(),x2=b.getX(),y2=b.getY(),z2=b.getZ();
        float w=w1*w2-x1*x2-y1*y2-z1*z2;
        float x=w1*x2+x1*w2+y1*z2-z1*y2;
        float y=w1*y2-x1*z2+y1*w2+z1*x2;
        float z=w1*z2+x1*y2-y1*x2+z1*w2;
        return new Quaternion(x,y,z,w);
    }
    // ============================================================================

    // ============================ retrievePath ============================
    public void retrievePath() {
        Log.i(TAG, "retrievePath: start");
        if (!pathSequence.checkItem(findItem)) {
            Log.i(TAG, "retrievePath: item not recorded, skip");
            return;
        }
        boolean first=true;
        Quaternion tgtQ=null; Point tgtP=null;
        while (pathSequence.size()>0) {
            Path p = pathSequence.retrievePath();
            Log.i(TAG, "recordItem: " + p.item);

            if (first) { tgtQ=p.quaternion; first=false; }
            if (p.item.equals(findItem)) {
                tgtP=p.point; tgtQ=p.quaternion;
                api.moveTo(tgtP,tgtQ,false);
                Log.i(TAG, "retrievePath: match found, stop");
                break;
            }
            api.moveTo(p.point,tgtQ,false);
        }
        Log.i(TAG, "retrievePath: end");
    }
    // ============================================================================

    private void takeSnapshotEarly() {
        api.takeTargetItemSnapshot();
    }

    private void saveBitmapIfDebug(Bitmap bmp, String filename) {
        if (DEBUG || (SAVE_PROCESSED_IMAGE && filename.startsWith("_processed"))) {
            api.saveBitmapImage(bmp, filename);
        }
    }

    private void saveMatIfDebug(Mat mat, String name) {
        if (DEBUG || SAVE_PROCESSED_IMAGE) {
            api.saveMatImage(mat, name);
        }
    }
}

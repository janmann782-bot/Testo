package space.aurelia.kefirvtuber;

import android.graphics.PointF;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceLandmark;


/** Immutable tracking data. Never stores camera images or face identity. */
public final class Pose {
    public final float yaw, roll, pitch, x, y, eyeLeft, eyeRight, mouth, smile;
    public final long timeMillis;

    public Pose(float yaw, float roll, float pitch, float x, float y,
                float eyeLeft, float eyeRight, float mouth, float smile) {
        this.yaw=yaw; this.roll=roll; this.pitch=pitch;
        this.x=x; this.y=y; this.eyeLeft=eyeLeft; this.eyeRight=eyeRight;
        this.mouth=mouth; this.smile=smile;
        timeMillis=android.os.SystemClock.uptimeMillis();
    }

    public static Pose neutral() { return new Pose(0,0,0,0,0,1,1,0,0); }

    public static Pose fromFace(Face face, int imageWidth, int imageHeight) {
        float yaw = clamp(face.getHeadEulerAngleY() / 36f, -1, 1);
        float roll = clamp(face.getHeadEulerAngleZ() / 31f, -1, 1);
        float pitch = clamp(face.getHeadEulerAngleX() / 27f, -1, 1);
        float fx = clamp((face.getBoundingBox().exactCenterX() / Math.max(1,imageWidth) - .5f)*2f,-1,1);
        float fy = clamp((face.getBoundingBox().exactCenterY() / Math.max(1,imageHeight)-.5f)*2f,-1,1);
        Float l = face.getLeftEyeOpenProbability();
        Float r = face.getRightEyeOpenProbability();
        Float s = face.getSmilingProbability();
        float left  = l == null ? 1f : clamp((l-.18f)*1.8f,0,1);
        float right = r == null ? 1f : clamp((r-.18f)*1.8f,0,1);
        float smile = s == null ? 0 : clamp(s,0,1);
        float open = estimateMouth(face);
        return new Pose(yaw, roll, pitch, fx,fy,left,right,open,smile);
    }

    private static float estimateMouth(Face f) {
        // Use three sparse landmarks rather than the much heavier per-frame
        // lip contours; microphone input remains available for more precise speech.
        FaceLandmark left=f.getLandmark(FaceLandmark.MOUTH_LEFT);
        FaceLandmark right=f.getLandmark(FaceLandmark.MOUTH_RIGHT);
        FaceLandmark bottom=f.getLandmark(FaceLandmark.MOUTH_BOTTOM);
        if(left==null||right==null||bottom==null)return 0f;
        float corners=(left.getPosition().y+right.getPosition().y)*.5f;
        float relative=(bottom.getPosition().y-corners)/Math.max(1f,f.getBoundingBox().height());
        return clamp((relative-.025f)*18f,0f,1f);
    }
    public static float clamp(float x,float a,float b) { return Math.max(a,Math.min(b,x)); }
}

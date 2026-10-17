package space.aurelia.kefirvtuber;

import android.Manifest;
import androidx.activity.ComponentActivity;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.util.Size;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import java.util.concurrent.atomic.AtomicBoolean;
import java.io.OutputStream;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import android.opengl.GLSurfaceView;

/** Fullscreen 3D VTuber. Face images never leave the phone or enter the rendered scene. */
public final class MainActivity extends ComponentActivity {
    private static final int CAMERA_PERMISSION=501, MIC_PERMISSION=502;
    private FrameLayout root,stage;
    private LinearLayout bar;
    private TextView ghost,micButton,cameraButton;
    private GLSurfaceView surface;
    private KefirRenderer renderer;
    private FrameRatio ratio=FrameRatio.PORTRAIT;
    private boolean uiVisible=true,trackingEnabled=true,mirror=true,micOn=false;
    private FaceDetector faceDetector;
    private ProcessCameraProvider cameraProvider;
    private ExecutorService cameraExecutor,ioExecutor;
    private final AtomicBoolean faceBusy=new AtomicBoolean(false);
    private ImageAnalysis currentAnalysis;
    private volatile boolean microphoneRunning=false;
    private AudioRecord recorder;
    private Thread audioThread;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setStatusBarColor(Color.BLACK);getWindow().setNavigationBarColor(Color.BLACK);
        fullscreen();
        cameraExecutor=Executors.newSingleThreadExecutor();ioExecutor=Executors.newSingleThreadExecutor();
        FaceDetectorOptions options=new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .setMinFaceSize(.12f).build();
        faceDetector=FaceDetection.getClient(options);
        makeUi();
        if(!hasCamera())requestPermissions(new String[]{Manifest.permission.CAMERA},CAMERA_PERMISSION);
    }
    private void fullscreen(){
        getWindow().getDecorView().setSystemUiVisibility(5894|View.SYSTEM_UI_FLAG_LAYOUT_STABLE|View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION|View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
    }
    private static int dp(Context c,float n){return (int)(n*c.getResources().getDisplayMetrics().density+.5f);}
    private void makeUi(){
        root=new FrameLayout(this);root.setBackgroundColor(Color.BLACK);
        surface=new GLSurfaceView(this);surface.setEGLContextClientVersion(2);
        surface.setPreserveEGLContextOnPause(true);
        renderer=new KefirRenderer(this,this::savePng);
        surface.setRenderer(renderer);surface.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        stage=new FrameLayout(this);stage.setBackgroundColor(Color.rgb(10,14,21));
        stage.addView(surface,new FrameLayout.LayoutParams(-1,-1));
        root.addView(stage,new FrameLayout.LayoutParams(-1,-1,Gravity.CENTER));
        bar=new LinearLayout(this);bar.setGravity(Gravity.CENTER);bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setPadding(dp(this,3),dp(this,3),dp(this,3),dp(this,3));bar.setBackground(panel());
        TextView b16=action("16:9",51,()->switchRatio(FrameRatio.LANDSCAPE));
        TextView b9=action("9:16",51,()->switchRatio(FrameRatio.PORTRAIT));
        TextView b4=action("4:3",51,()->switchRatio(FrameRatio.CLASSIC));
        cameraButton=action("◎",39,()->{trackingEnabled=!trackingEnabled;
            if(trackingEnabled){if(hasCamera())startCamera();else requestPermissions(new String[]{Manifest.permission.CAMERA},CAMERA_PERMISSION);}else stopCamera();
            cameraButton.setAlpha(trackingEnabled?1f:.40f);
        });
        micButton=action("◖",39,()->toggleMicrophone());
        TextView flip=action("⇄",39,()->{mirror=!mirror;renderer.setMirrored(mirror);});
        TextView png=action("▣",39,()->renderer.capture());
        bar.addView(b16);bar.addView(b9);bar.addView(b4);
        bar.addView(cameraButton);bar.addView(micButton);bar.addView(flip);bar.addView(png);micButton.setAlpha(.45f);
        HorizontalScrollView scroller=new HorizontalScrollView(this);scroller.setHorizontalScrollBarEnabled(false);
        scroller.addView(bar,new ViewGroup.LayoutParams(-2,-2));
        FrameLayout.LayoutParams bp=new FrameLayout.LayoutParams(-2,-2,Gravity.BOTTOM|Gravity.CENTER_HORIZONTAL);
        bp.bottomMargin=dp(this,16);root.addView(scroller,bp);
        // Keep the low-profile strip as one controllable overlay.
        controls=scroller;
        ghost=new TextView(this);ghost.setText("◌");ghost.setGravity(Gravity.CENTER);
        ghost.setTextSize(29);ghost.setTextColor(Color.WHITE);ghost.setAlpha(.24f);
        ghost.setBackground(panel());ghost.setOnClickListener(v->{uiVisible=!uiVisible;controls.setVisibility(uiVisible?View.VISIBLE:View.GONE);
            ghost.setAlpha(uiVisible?.24f:.15f);});
        FrameLayout.LayoutParams gp=new FrameLayout.LayoutParams(dp(this,42),dp(this,42),Gravity.TOP|Gravity.RIGHT);
        gp.topMargin=dp(this,14);gp.rightMargin=dp(this,11);
        root.addView(ghost,gp);
        setContentView(root);
        root.addOnLayoutChangeListener((v,a,b,c,d,e,f,g,h)->updateStageSize());
        updateStageSize();
    }
    private View controls;
    private TextView action(String symbol,int width,Runnable run){
        TextView v=new TextView(this);v.setText(symbol);v.setTextColor(Color.rgb(237,239,239));
        v.setTypeface(Typeface.DEFAULT,Typeface.BOLD);v.setTextSize(symbol.contains(":")?12:22);
        v.setGravity(Gravity.CENTER);v.setBackground(buttonBackground());v.setOnClickListener(view->run.run());
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(dp(this,width),dp(this,39));
        lp.leftMargin=dp(this,2);lp.rightMargin=dp(this,2);v.setLayoutParams(lp);return v;
    }
    private GradientDrawable panel(){GradientDrawable bg=new GradientDrawable();bg.setColor(0x87313B45);bg.setCornerRadius(dp(this,20));return bg;}
    private GradientDrawable buttonBackground(){GradientDrawable bg=new GradientDrawable();bg.setColor(0x493F4D59);bg.setCornerRadius(dp(this,13));return bg;}
    private void switchRatio(FrameRatio next){
        ratio=next;
        // Rotate the Android activity itself, not merely the preview rectangle.
        int orientation=next==FrameRatio.PORTRAIT
            ?ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE;
        if(getRequestedOrientation()!=orientation)setRequestedOrientation(orientation);
        updateStageSize();
    }
    @Override public void onConfigurationChanged(Configuration configuration){
        super.onConfigurationChanged(configuration);
        updateStageSize();
        if(currentAnalysis!=null){
            currentAnalysis.setTargetRotation(getWindowManager().getDefaultDisplay().getRotation());
        }
    }
    private void updateStageSize(){
        if(root==null||stage==null)return;
        int w=root.getWidth(),h=root.getHeight();if(w<=0||h<=0)return;
        float a=ratio.aspect();
        int sw=w,sh=Math.round(w/a);
        if(sh>h){sh=h;sw=Math.round(h*a);}
        FrameLayout.LayoutParams p=new FrameLayout.LayoutParams(sw,sh,Gravity.CENTER);
        stage.setLayoutParams(p);
    }
    private boolean hasCamera(){return checkSelfPermission(Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED;}
    private boolean hasMic(){return checkSelfPermission(Manifest.permission.RECORD_AUDIO)==PackageManager.PERMISSION_GRANTED;}
    private void startCamera(){
        if(!trackingEnabled||!hasCamera())return;
        ListenableFuture<ProcessCameraProvider> future=ProcessCameraProvider.getInstance(this);
        future.addListener(()->{
            try{cameraProvider=future.get();bindCamera();}
            catch(Exception e){android.util.Log.e("KefirVTuber","CameraX start",e);toast("Камера недоступна");}
        },ContextCompat.getMainExecutor(this));
    }
    private void bindCamera(){
        if(cameraProvider==null||!trackingEnabled)return;
        cameraProvider.unbindAll();
        ImageAnalysis stream=new ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetResolution(new Size(480,360))
            .setTargetRotation(getWindowManager().getDefaultDisplay().getRotation())
            .build();
        currentAnalysis=stream;
        stream.setAnalyzer(cameraExecutor,this::analyzeFrame);
        cameraProvider.bindToLifecycle(this,CameraSelector.DEFAULT_FRONT_CAMERA,stream);
    }
    private void analyzeFrame(ImageProxy frame){
        if(!trackingEnabled || !faceBusy.compareAndSet(false,true)){frame.close();return;}
        android.media.Image media=frame.getImage();
        if(media==null){faceBusy.set(false);frame.close();return;}
        int rotation=frame.getImageInfo().getRotationDegrees();
        InputImage image=InputImage.fromMediaImage(media,rotation);
        int w=frame.getWidth(),h=frame.getHeight();
        if(rotation==90||rotation==270){int t=w;w=h;h=t;}
        final int rw=w,rh=h;
        faceDetector.process(image)
            .addOnSuccessListener(cameraExecutor, faces->{
                if(!faces.isEmpty()){
                    Face closest=faces.get(0);for(Face f:faces){if(f.getBoundingBox().width()>closest.getBoundingBox().width())closest=f;}
                    renderer.updateFace(Pose.fromFace(closest,rw,rh));
                }
            }).addOnFailureListener(cameraExecutor,e->android.util.Log.w("KefirVTuber","Face frame failed",e))
            .addOnCompleteListener(cameraExecutor,task->{faceBusy.set(false);frame.close();});
    }
    private void stopCamera(){
        if(cameraProvider!=null)cameraProvider.unbindAll();
        currentAnalysis=null;
        renderer.updateFace(Pose.neutral());
    }
    private void toggleMicrophone(){
        if(micOn){stopMicrophone();return;}
        if(!hasMic()){requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},MIC_PERMISSION);return;}
        startMicrophone();
    }
    private void startMicrophone(){
        if(micOn||!hasMic())return;
        final int hz=16000;
        int min=AudioRecord.getMinBufferSize(hz,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);
        if(min<=0){toast("Микрофон недоступен");return;}
        try{
            recorder=new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,hz,AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,Math.max(4096,min*2));
            if(recorder.getState()!=AudioRecord.STATE_INITIALIZED){recorder.release();recorder=null;return;}
            recorder.startRecording();microphoneRunning=true;micOn=true;renderer.setMicEnabled(true);micButton.setAlpha(1);
            audioThread=new Thread(()->{
                short[] pcm=new short[1024];float previous=0;
                while(microphoneRunning){
                    AudioRecord input=recorder;if(input==null)break;
                    int n;try{n=input.read(pcm,0,pcm.length);}catch(IllegalStateException ex){break;}
                    if(n>0){double power=0;for(int i=0;i<n;i++)power+=(double)pcm[i]*pcm[i];
                        float rms=(float)Math.sqrt(power/n)/32768f;
                        float level=Pose.clamp((rms-.006f)*18f,0,1);previous=.15f*previous+.85f*level;
                        renderer.setMicLevel(previous);
                    }
                }
            },"Aurelia-VTuber-Audio");audioThread.start();
        }catch(SecurityException|IllegalStateException e){toast("Микрофон недоступен");}
    }
    private void stopMicrophone(){
        micOn=false;microphoneRunning=false;renderer.setMicEnabled(false);renderer.setMicLevel(0);
        if(micButton!=null)micButton.setAlpha(.45f);
        AudioRecord rec=recorder;recorder=null;
        if(rec!=null){try{rec.stop();}catch(Exception ignored){}rec.release();}
    }
    private void savePng(Bitmap bitmap){
        ioExecutor.execute(()->{
            Uri uri=null;
            try{
                String name="Kefirstan_"+System.currentTimeMillis()+"_"+ratio.label.replace(':','x')+".png";
                ContentValues cv=new ContentValues();cv.put(MediaStore.Images.Media.DISPLAY_NAME,name);
                cv.put(MediaStore.Images.Media.MIME_TYPE,"image/png");
                cv.put(MediaStore.Images.Media.RELATIVE_PATH,Environment.DIRECTORY_PICTURES+"/Aurelia");
                cv.put(MediaStore.Images.Media.IS_PENDING,1);
                uri=getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,cv);
                if(uri==null)throw new IllegalStateException("MediaStore insert failed");
                try(OutputStream out=getContentResolver().openOutputStream(uri)){
                    if(out==null||!bitmap.compress(Bitmap.CompressFormat.PNG,100,out))throw new IllegalStateException("PNG encode failed");
                }
                cv.clear();cv.put(MediaStore.Images.Media.IS_PENDING,0);
                getContentResolver().update(uri,cv,null,null);
                runOnUiThread(()->toast("PNG сохранен"));
            }catch(Exception e){android.util.Log.e("KefirVTuber","PNG",e);runOnUiThread(()->toast("Не удалось сохранить PNG"));}
            finally{bitmap.recycle();}
        });
    }
    private void toast(String message){Toast.makeText(this,message,Toast.LENGTH_SHORT).show();}
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] grants){
        super.onRequestPermissionsResult(request,permissions,grants);
        if(grants.length>0&&grants[0]==PackageManager.PERMISSION_GRANTED){
            if(request==CAMERA_PERMISSION)startCamera();else if(request==MIC_PERMISSION)startMicrophone();
        }
    }
    @Override protected void onResume(){super.onResume();fullscreen();if(surface!=null)surface.onResume();if(hasCamera()&&trackingEnabled)startCamera();}
    @Override protected void onPause(){stopMicrophone();stopCamera();if(surface!=null)surface.onPause();super.onPause();}
    @Override protected void onDestroy(){stopMicrophone();stopCamera();if(faceDetector!=null)faceDetector.close();
        if(cameraExecutor!=null)cameraExecutor.shutdown();if(ioExecutor!=null)ioExecutor.shutdown();super.onDestroy();}
}

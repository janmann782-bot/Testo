package space.aurelia.kefirvtuber;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.opengl.Matrix;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.opengl.GLSurfaceView;
import android.os.SystemClock;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/** Real 3D model renderer: lit OBJ office + 3D countryball rig, never camera-image compositing. */
final class KefirRenderer implements GLSurfaceView.Renderer {
    interface SnapshotListener { void onSnapshot(Bitmap bitmap); }
    private final Context context;
    private final SnapshotListener listener;
    private OfficeModel office;
    private Mesh3D head,facePatch,eyeOutline,eyeWhite,mouthBlack,cube;
    private int program,attributePosition,attributeNormal,attributeUv,uniformMvp,uniformModel,uniformColor,uniformTexture,uniformSampler,uniformShadow;
    private final float[] proj=new float[16],view=new float[16],vp=new float[16],base=new float[16];
    private final float[] model=new float[16],tmp=new float[16],mvp=new float[16];
    private final Map<String,Integer> textures=new HashMap<>();
    private int width=1,height=1;
    private volatile Pose requested=Pose.neutral();
    private Pose smooth=Pose.neutral();
    private volatile boolean snap=false;
    private volatile float micLevel=0;
    private volatile boolean audioEnabled=false;
    private volatile boolean mirror=true;
    private long lastFrameNanos=0;
    KefirRenderer(Context context,SnapshotListener listener){this.context=context.getApplicationContext();this.listener=listener;}
    void updateFace(Pose p){requested=p;}
    void setMicLevel(float value){micLevel=Pose.clamp(value,0,1);}
    void setMicEnabled(boolean value){audioEnabled=value;}
    void setMirrored(boolean value){mirror=value;}
    void capture(){snap=true;}
    @Override public void onSurfaceCreated(GL10 gl,EGLConfig config){
        GLES20.glClearColor(.145f,.196f,.270f,1);
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);GLES20.glDepthFunc(GLES20.GL_LEQUAL);
        GLES20.glDisable(GLES20.GL_CULL_FACE);
        program=compileProgram(VERTEX,FRAGMENT);
        attributePosition=GLES20.glGetAttribLocation(program,"aPos");
        attributeNormal=GLES20.glGetAttribLocation(program,"aNormal");
        attributeUv=GLES20.glGetAttribLocation(program,"aUv");
        uniformMvp=GLES20.glGetUniformLocation(program,"uMvp");
        uniformModel=GLES20.glGetUniformLocation(program,"uModel");
        uniformColor=GLES20.glGetUniformLocation(program,"uColor");
        uniformTexture=GLES20.glGetUniformLocation(program,"uTextured");
        uniformShadow=GLES20.glGetUniformLocation(program,"uShadow");
        uniformSampler=GLES20.glGetUniformLocation(program,"uSampler");
        GLES20.glUseProgram(program);
        GLES20.glEnableVertexAttribArray(attributePosition);
        GLES20.glEnableVertexAttribArray(attributeNormal);
        GLES20.glEnableVertexAttribArray(attributeUv);
        GLES20.glUniform1i(uniformSampler,0);
        for(String name:new String[]{"flag_kefiristan","flag_kefiristan_wall","flag_kefiria","flag_tvorograd","flag_molozori"}){
            try {textures.put(name,loadTexture(name+".png"));}catch(Exception ignored){}
        }
        try{office=new OfficeModel(context.getAssets());}catch(Exception e){android.util.Log.e("KefirVTuber","3D office OBJ loading failed",e);}
        head=Mesh3D.countryball();cube=Mesh3D.unitCube();
        // A small smooth face patch hides the harsh white X intersection
        // that previously read as a gigantic nose between the eyes.
        facePatch=Mesh3D.discOnBall(0,0,.40f,.46f,.020f,96);
        eyeOutline=Mesh3D.discOnBall(0,0,.215f,.255f,.050f,68);
        eyeWhite=Mesh3D.discOnBall(0,0,.178f,.218f,.073f,68);
        mouthBlack=Mesh3D.discOnBall(0,0,.14f,.10f,.065f,68);
    }
    @Override public void onSurfaceChanged(GL10 gl,int width,int height){
        this.width=Math.max(1,width);this.height=Math.max(1,height);
        GLES20.glViewport(0,0,this.width,this.height);
        calculateCamera();
    }
    private void calculateCamera(){
        float a=(float)width/height;
        float fov=a<.85f?48f:(a>1.55f?42f:45f);
        float cameraDistance=a<.85f?4.0f:(a>1.55f?3.72f:4.05f);
        float camY=a<.85f?3.40f:(a>1.55f?3.68f:3.58f);
        float focalZ=a>1.55f?-7.02f:-7.15f;
        Matrix.perspectiveM(proj,0,fov,a,.12f,80f);
        Matrix.setLookAtM(view,0,0,camY,cameraDistance,0,3.02f,focalZ,0,1,0);
        Matrix.multiplyMM(vp,0,proj,0,view,0);
        Matrix.setIdentityM(base,0);
    }
    @Override public void onDrawFrame(GL10 gl){
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT|GLES20.GL_DEPTH_BUFFER_BIT);
        GLES20.glUseProgram(program);
        long now=SystemClock.elapsedRealtimeNanos();
        float dt=lastFrameNanos==0?1f/60f:Math.min(.07f,(now-lastFrameNanos)/1_000_000_000f);
        lastFrameNanos=now;
        Pose target=requested;
        if(SystemClock.uptimeMillis()-target.timeMillis>950)target=Pose.neutral();
        float t=1f-(float)Math.exp(-dt*38f);
        smooth=interpolate(smooth,target,t);
        drawOffice();
        drawCountryball(now*.000000001f);
        if(snap){snap=false;captureFrame();}
    }
    private static Pose interpolate(Pose p,Pose q,float t){
        return new Pose(mix(p.yaw,q.yaw,t),mix(p.roll,q.roll,t),mix(p.pitch,q.pitch,t),
          mix(p.x,q.x,t),mix(p.y,q.y,t),mix(p.eyeLeft,q.eyeLeft,t),mix(p.eyeRight,q.eyeRight,t),
          mix(p.mouth,q.mouth,t),mix(p.smile,q.smile,t));
    }
    private static float mix(float a,float b,float t){return a+(b-a)*t;}
    private void drawOffice(){
        if(office==null)return;
        // Opaque geometry first, translucent penumbra geometry after.
        for(OfficeModel.Part p:office.parts){
            if(p.material.startsWith("shadow_"))continue;
            float[] rgba=matColor(p.material);
            draw(p.mesh,base,rgba,p.material.startsWith("flag_")?p.material:null);
        }
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA,GLES20.GL_ONE_MINUS_SRC_ALPHA);
        GLES20.glDepthMask(false);
        for(OfficeModel.Part p:office.parts){
            if(p.material.startsWith("shadow_")){
                float[] shadowMatrix=base;
                if(p.material.equals("shadow_ball")){
                    // Floating ball casts a soft moving patch on the desk.
                    shadowMatrix=base.clone();
                    Matrix.translateM(shadowMatrix,0,smooth.x*.20f,0,0);
                }
                draw(p.mesh,shadowMatrix,new float[]{1,1,1,1},null,true);
            }
        }
        GLES20.glDepthMask(true);
        GLES20.glDisable(GLES20.GL_BLEND);
    }
    private void drawCountryball(float now){
        // Countryball rests directly on the desk. No body, neck, shirt, tie, or hair.
        float x=smooth.x*.20f, y=smooth.y*-.10f;
        Matrix.setIdentityM(model,0);
        Matrix.translateM(model,0,x,3.18f+y,-7.08f);
        Matrix.rotateM(model,0,(mirror?-1:1)*smooth.roll*23f,0,0,1);
        Matrix.rotateM(model,0,smooth.pitch*-25f,1,0,0);
        Matrix.rotateM(model,0,(mirror?-1:1)*smooth.yaw*39f,0,1,0);
        float[] headMat=model.clone();
        draw(head,headMat,new float[]{1,1,1,1},"flag_kefiristan");
        // Flush surface, not a protruding 3D nose.
        float[] mask=headMat.clone();
        float maskY=-.05f;
        Matrix.translateM(mask,0,0,maskY,(float)Math.sqrt(1-maskY*maskY));
        draw(facePatch,mask,color("#ffffff"),null);
        eye(headMat,-.30f,.14f,Math.max(.02f,smooth.eyeLeft));
        eye(headMat,.30f,.14f,Math.max(.02f,smooth.eyeRight));
        float opening=Pose.clamp(Math.max(smooth.mouth,audioEnabled?micLevel:0),0,1);
        float mouthY=-.46f;
        float[] mouthMatrix=headMat.clone();
        Matrix.translateM(mouthMatrix,0,0,mouthY,(float)Math.sqrt(1-mouthY*mouthY));
        scaleAndDraw(mouthBlack,mouthMatrix,1f,.22f+opening*.94f,1f,color("#0a090b"));
    }
    private void eye(float[] headMat,float x,float y,float openness){
        float z=(float)Math.sqrt(1-x*x-y*y);
        float[] m=headMat.clone(); Matrix.translateM(m,0,x,y,z);
        scaleAndDraw(eyeOutline,m,1,openness,1,color("#08090c"));
        float[] n=headMat.clone();Matrix.translateM(n,0,x,y,z);
        scaleAndDraw(eyeWhite,n,1,openness,1,color("#f9faf7"));
    }
    private void cubeAt(float[] parent,float x,float y,float z,float w,float h,float d,float[] col){
        float[] m=parent.clone();Matrix.translateM(m,0,x,y,z);scaleAndDraw(cube,m,w,h,d,col);
    }
    private void scaleAndDraw(Mesh3D mesh,float[] parent,float sx,float sy,float sz,float[] col){
        float[] m=parent.clone();Matrix.scaleM(m,0,sx,sy,sz);draw(mesh,m,col,null);
    }
    private void draw(Mesh3D mesh,float[] mat,float[] col,String tex){draw(mesh,mat,col,tex,false);}
    private void draw(Mesh3D mesh,float[] mat,float[] col,String tex,boolean shadow){
        if(mesh==null)return;
        Matrix.multiplyMM(mvp,0,vp,0,mat,0);
        GLES20.glUniformMatrix4fv(uniformMvp,1,false,mvp,0);
        GLES20.glUniformMatrix4fv(uniformModel,1,false,mat,0);
        GLES20.glUniform4fv(uniformColor,1,col,0);
        Integer texture=tex==null?null:textures.get(tex);
        GLES20.glUniform1f(uniformTexture,texture==null?0:1);
        GLES20.glUniform1f(uniformShadow,shadow?1f:0f);
        if(texture!=null){GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,texture);}
        mesh.draw(attributePosition,attributeNormal,attributeUv);
    }
    private int loadTexture(String asset) throws Exception {
        Bitmap original;
        try(java.io.InputStream in=context.getAssets().open(asset)) {original=BitmapFactory.decodeStream(in);}
        if(original==null) throw new IllegalStateException(asset);
        // Blender UV uses upper edge V=1. OpenGL's bitmap rows are inverted on upload.
        Bitmap flipped=Bitmap.createBitmap(original.getWidth(),original.getHeight(),Bitmap.Config.ARGB_8888);
        Canvas c=new Canvas(flipped);android.graphics.Matrix mm=new android.graphics.Matrix();mm.postScale(1,-1);mm.postTranslate(0,original.getHeight());c.drawBitmap(original,mm,null);
        int[] id=new int[1];GLES20.glGenTextures(1,id,0);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,id[0]);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE);
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D,0,flipped,0);
        flipped.recycle();original.recycle();return id[0];
    }
    private void captureFrame(){
        try {
            ByteBuffer bytes=ByteBuffer.allocateDirect(width*height*4).order(ByteOrder.nativeOrder());
            GLES20.glReadPixels(0,0,width,height,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,bytes);
            int[] src=new int[width*height],dst=new int[src.length];bytes.asIntBuffer().get(src);
            for(int y=0;y<height;y++) for(int x=0;x<width;x++) {
                int abgr=src[y*width+x];
                dst[(height-1-y)*width+x]=(abgr&0xff00ff00)|((abgr&0xff)<<16)|((abgr>>16)&0xff);
            }
            Bitmap bitmap=Bitmap.createBitmap(dst,width,height,Bitmap.Config.ARGB_8888);
            if(listener!=null)listener.onSnapshot(bitmap);
        }catch(Throwable e){android.util.Log.e("KefirVTuber","PNG capture error",e);}
    }
    private static float[] color(String hex){
        int v=android.graphics.Color.parseColor(hex);
        return new float[]{android.graphics.Color.red(v)/255f,android.graphics.Color.green(v)/255f,android.graphics.Color.blue(v)/255f,1};
    }
    private static float[] matColor(String n){
        switch(n){
            case "oval_material_1":return color("#b8c0c6");
            case "oval_material_2":return color("#203544");
            case "oval_material_3":return color("#534236");
            case "oval_material_4":return color("#9f4c4c");
            case "oval_material_5":return color("#2b2520");
            case "oval_material_6":return color("#040000");
            case "oval_material_7":return color("#20385e");
            case "oval_material_8":return color("#ffffff");
            case "finial_gold":return color("#be9c62");
            case "wall":return color("#2b435c");
            case "wall_side":return color("#31475e");
            case "molding":return color("#a18760");
            case "molding_muted":return color("#6c7f95");
            case "floor":return color("#403834");
            case "wood":return color("#74584b");
            case "dark_wood":return color("#342b29");
            case "desk_top":return color("#6f574d");
            case "brass":return color("#b6975f");
            case "pole_steel":return color("#9da6ad");
            case "base_brass":return color("#ad9c66");
            case "sleeve":return color("#dad5c6");
            case "frame_gold":return color("#c2a775");
            case "frame_dark":return color("#121920");
            case "paper":return color("#d5c9ac");
            case "paper_white":return color("#ede5d2");
            case "leather":return color("#1d2b36");
            case "leather_light":return color("#34505e");
            case "mic_dark":return color("#191b1f");
            case "steel":return color("#75838d");
            case "book_blue":return color("#4f6f8f");
            case "book_red":return color("#7d6166");
            case "book_green":return color("#5b7e7a");
            case "book_tan":return color("#b49e7e");
            case "rug":return color("#8a755c");
            case "rug_inner":return color("#324451");
            case "light":return color("#e2d5a7");
            case "glass":return color("#82999d");
            default:return color("#ffffff");
        }
    }
    private static int shader(int type,String source){
        int s=GLES20.glCreateShader(type);GLES20.glShaderSource(s,source);GLES20.glCompileShader(s);
        int[] ok=new int[1];GLES20.glGetShaderiv(s,GLES20.GL_COMPILE_STATUS,ok,0);
        if(ok[0]==0)throw new IllegalStateException("GL shader: "+GLES20.glGetShaderInfoLog(s));
        return s;
    }
    private static int compileProgram(String vertex,String fragment){
        int p=GLES20.glCreateProgram();
        GLES20.glAttachShader(p,shader(GLES20.GL_VERTEX_SHADER,vertex));
        GLES20.glAttachShader(p,shader(GLES20.GL_FRAGMENT_SHADER,fragment));
        GLES20.glLinkProgram(p);int[] ok=new int[1];GLES20.glGetProgramiv(p,GLES20.GL_LINK_STATUS,ok,0);
        if(ok[0]==0)throw new IllegalStateException("GL link: "+GLES20.glGetProgramInfoLog(p));
        return p;
    }
    // GLES 2.0 diffuse key light with generous ambient fill; soft shadow
    // decals use radial falloff in the fragment shader, no bitmap rendering.
    private static final String VERTEX=
        "attribute vec3 aPos; attribute vec3 aNormal; attribute vec2 aUv;"+
        "uniform mat4 uMvp; uniform mat4 uModel; varying vec2 vUv; varying float vLight;"+
        "void main(){vec3 n=normalize(mat3(uModel)*aNormal);"+
        "vec3 key=normalize(vec3(-0.38,0.83,0.36));"+
        "vLight=0.79+0.21*max(dot(n,key),0.0);"+
        "vUv=aUv;gl_Position=uMvp*vec4(aPos,1.0);}";
    private static final String FRAGMENT=
        "precision mediump float;uniform vec4 uColor;uniform float uTextured;"+
        "uniform float uShadow;uniform sampler2D uSampler;varying vec2 vUv;varying float vLight;"+
        "void main(){if(uShadow>0.5){vec2 d=(vUv-vec2(.5))*2.0;float r=length(d);"+
        "float a=0.40*(1.0-smoothstep(0.12,1.0,r));"+
        "gl_FragColor=vec4(0.015,0.020,0.027,a);return;}"+
        "vec4 albedo=uColor; if(uTextured>0.5)albedo*=texture2D(uSampler,vUv);"+
        "gl_FragColor=vec4(albedo.rgb*vLight,albedo.a);}";
}

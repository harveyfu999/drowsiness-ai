package com.example.drowsinessdetector

import android.Manifest
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.pm.PackageManager
import android.media.MediaPlayer
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Looper
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.SurfaceView
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.snackbar.Snackbar
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.lang.Float.max
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class OpenCVText(val text: String, val pos: Point) {
    fun display(frame: Mat): Mat {
        Imgproc.putText(frame, text, pos,
            Imgproc.FONT_HERSHEY_PLAIN, 3.0, Scalar(255.0, 0.0, 0.0), 4)
        return frame
    }
}
class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private var eyes: Array<Rect>? = null
    private lateinit var camBridgeView: CameraBridgeViewBase
    private lateinit var stopMsgView: TextView

    private var showStopMsg: AtomicBoolean  = AtomicBoolean(false)
    private var showStopMsgLock = ReentrantLock()
    private var showStopMsgCondition = showStopMsgLock.newCondition()

    private lateinit var layout: View
    private lateinit var face_classfier: CascadeClassifier
    private lateinit var eye_classfier: CascadeClassifier
    private lateinit var ort_sess: OrtSession
    private var dispText: MutableList<OpenCVText> = mutableListOf()
    private var frame_count: Int = 0

    private var eyeThreshold: Float = 0.6F
    private var delayBeforeDetectionRising: Float = 2.0F // in seconds
    private var delayBeforeDetectionFalling: Float = 0.2F // in seconds
    private var delayBeforeDetectionLock: Lock = ReentrantLock()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        camBridgeView = findViewById(R.id.CameraView)
        layout = findViewById(R.id.MainLayout)
        stopMsgView = findViewById(R.id.stopMessage)
        stopMsgView.visibility = View.VISIBLE


        Thread {
            Looper.prepare()
            var lastChange: Long = 0
            var lastVal: Boolean = false
            val mediaPlayer = MediaPlayer.create(this, R.raw.alarm_facility)
            mediaPlayer.isLooping = true
            mediaPlayer.start()
            while (true) {
                showStopMsgLock.withLock {
                    showStopMsgCondition.await()
                }

                val showSM = showStopMsg.get()
                delayBeforeDetectionLock.withLock {
                    if (showSM && lastVal &&
                        (System.currentTimeMillis() - lastChange) > delayBeforeDetectionRising * 1000
                    ) {
                        Log.wtf("sleeping", "start playing")
                        mediaPlayer.start()
                    }

                    if (!lastVal && !showSM &&
                        (System.currentTimeMillis() - lastChange) > delayBeforeDetectionFalling * 1000
                    ) {
                        mediaPlayer.pause()
                        Log.wtf("sleeping", "stop playing")
                    }
                }
                if (lastVal != showSM) {
                    lastChange = System.currentTimeMillis()
                }
                lastVal = showSM

                // I don't use locks because I deliberately want to smooth out the signal
                //Thread.sleep((delayBeforeDetection * 1000).toLong())
                //Log.wtf("sleeping", (delayBeforeDetection * 1000).toLong().toString())
                //if (showStopMsg.get()) {
                //    Toast.makeText(this, "Wake UP", Toast.LENGTH_LONG).show()
                //    Log.wtf("sleeping", "wakeup")
                //    if (!mediaPlayer.isPlaying) {
                //        mediaPlayer.start()
                //    }
                //} else {
                //    mediaPlayer.pause()
                //}
            }
        }.start()

        val model_res = resources.openRawResource(R.raw.final_model_basic).readBytes()
        ort_sess = OrtEnvironment.getEnvironment().createSession(model_res)
        checkPermission()

        camBridgeView.visibility = SurfaceView.VISIBLE
        camBridgeView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT)
        camBridgeView.setCvCameraViewListener(this@MainActivity)

    }

    private fun checkPermission(): Boolean {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
                Snackbar.make(layout, "Please let me use your camera", Snackbar.LENGTH_LONG).show()
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
                camBridgeView.setCameraPermissionGranted()
            }
            return false
        } else {
            camBridgeView.setCameraPermissionGranted()
            return true
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Toast.makeText(this@MainActivity, "camera started", Toast.LENGTH_LONG).show()
    }

    override fun onCameraViewStopped() {
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        var frame = inputFrame!!.rgba()
        if (!this::face_classfier.isInitialized) {
            return frame
        }
        if (!this::eye_classfier.isInitialized) {
            return frame
        }

        if (frame_count % 10 == 0) {
            dispText = mutableListOf()
            val grayscale: Mat = Mat()
            val face = MatOfRect()
            Imgproc.cvtColor(frame, grayscale, Imgproc.COLOR_RGBA2GRAY)
            face_classfier.detectMultiScale(grayscale, face, 1.1, 2)

            val faces = face.toArray()
            faces?.let {
                if (it.size == 0) {
                    showStopMsg.set(false)
                }
            }

            Log.wtf("opencv-debug", faces?.size.toString())
            faces?.maxByOrNull {
                it.area()
            }?.let { face ->
                Imgproc.rectangle(frame, face.tl(), face.br(), Scalar(255.0, 0.0, 0.0), 8)
                val dx = face.br().x - face.tl().x
                val dy = face.br().y - face.tl().y
                //val roi = Rect(face.tl().x.toInt(), face.tl().y.toInt(), dx.toInt(), dy.toInt())
                val roi = Rect(face.tl(), face.br())
                val face_subimg = grayscale.submat(roi)

                val eye = MatOfRect()
                eye_classfier.detectMultiScale(face_subimg, eye, 1.2, 4)
                eyes = eye.toArray()
                eyes?.let {
                    showStopMsg.set(it.size < 2)

                    var eyes_copy: List<Rect> = it.map { x -> Rect(x.tl() + roi.tl(), x.size() ) }
                    eyes_copy = eyes_copy.map { it2 ->
                        val mid_x = (it2.tl().x + it2.br().x)/2
                        val mid_y = (it2.tl().y + it2.br().y)/2
                        val diff = kotlin.math.min(it2.br().x - it2.tl().x, 96.0/2)
                        Rect(Point(mid_x-diff, mid_y-diff), Point(mid_x+diff, mid_y+diff))
                    }
                    eyes_copy = eyes_copy.filter { it2 ->
                        //val emid_x = (it2.tl().x + it2.br().x)/2
                        val emid_y = (it2.tl().y + it2.br().y)/2

                        val fmid_y = (roi.tl().y + roi.br().y)/2
                        emid_y < fmid_y
                    }

                    eyes = eyes_copy.toTypedArray()
                }


                eyes?.map { eye_roi ->
                    val eye_img = grayscale.submat(eye_roi)

                    Imgproc.resize(eye_img, eye_img, Size(96.0, 96.0))
                    eye_img.convertTo(eye_img, CvType.CV_32F)

                    // TODO("use bytearray")
                    var float_arr = FloatArray(96*96*eye_img.channels())
                    eye_img.get(0, 0, float_arr)
                    float_arr = float_arr.map { 2.0F * it / 255 - 0.5F }.toFloatArray()

                    // 0.let {
                    //     val bb = ByteBuffer.allocate(4 * float_arr.size)
                    //     bb.order(ByteOrder.nativeOrder())
                    //     val fb = bb.asFloatBuffer()
                    //     fb.put(float_arr)
                    //     val byteArr = bb.array()
                    //     val bitmap = BitmapFactory.decodeByteArray(byteArr, 0, 96*96)
                    // }


                    var fb: FloatBuffer = FloatBuffer.allocate(96*96)
                    fb.put(float_arr)
                    fb.position(0)

                    val env = OrtEnvironment.getEnvironment()
                    val t = OnnxTensor.createTensor(env, fb, longArrayOf(1, 1, 96, 96))
                    val input_name = ort_sess.inputNames.iterator().next()
                    val results = ort_sess.run(mapOf(input_name to t))
                    results.map {
                        val resultTensor = it.value as OnnxTensor
                        Log.wtf("model-out", "out")
                        val out: Float = resultTensor.floatBuffer[0]

                        showStopMsg.set(showStopMsg.get() or (out < eyeThreshold))

                        dispText.add(
                            OpenCVText("${out.format(2)}", Point(eye_roi.tl().x-10, eye_roi.tl().y-10))
                        )
                        Log.wtf("model-out", out.toString())
                    }
                }
            }

            showStopMsgLock.withLock { showStopMsgCondition.signalAll() }
            if (showStopMsg.get()) {
                runOnUiThread(Runnable { stopMsgView.visibility = View.VISIBLE})
            } else {
                runOnUiThread(Runnable { stopMsgView.visibility = View.INVISIBLE})
            }
        }

        frame = dispText.fold(frame, { frame2, t -> t.display(frame2) })
        Log.wtf("opencv-debug", eyes?.size.toString())
        eyes?.let {
            for (eye in it) {
                Imgproc.rectangle(frame, eye.tl(), eye.br(), Scalar(0.0, 255.0, 0.0), 8)
            }
        }

        frame_count += 1
        return frame
    }

    fun initOpenCV() {
        //val files: Array<Pair<Resource, String>> = [Pair(R.raw.haarcascade_frontalface_alt2, "face.xml"),
        val files: Array<Pair<Int, String>> = arrayOf(
            Pair(R.raw.haarcascade_frontalface_default, "face.xml"),
            Pair(R.raw.haarcascade_eye, "eye.xml"),
        )
        for (file in files) {
            // initialize haarcascade
            val stream = resources.openRawResource(file.first)
            val cascade_dir = getDir("cascade_xml", MODE_PRIVATE)
            val cascade_file = File(cascade_dir, file.second)
            cascade_file.writeBytes(stream.readBytes())
            when (file.second) {
                "face.xml" -> face_classfier = CascadeClassifier(cascade_file.absolutePath)
                "eye.xml" -> eye_classfier = CascadeClassifier(cascade_file.absolutePath)
            }
        }
    }

    override fun onResume() {
        super.onResume()

        // val status = OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, MyBaseLoaderCb(camBridgeView, this@MainActivity))
        System.loadLibrary("opencv_java4")
        val status = OpenCVLoader.initDebug()
        if (status) {
            camBridgeView.enableView()
            camBridgeView.enableFpsMeter()
            Log.wtf("OpencvTag", "OpenCV loaded")
            Toast.makeText(this@MainActivity, "Loaded opencv", Toast.LENGTH_LONG).show()
            initOpenCV()
        } else {
            Toast.makeText(this@MainActivity, "Can't load opencv", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        camBridgeView.disableView()
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.settings_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.eyeThreshold -> {
                val dialog = EyeThresholdDialog({ eyeThreshold = it }, R.id.editEyeThreshold, "Select eye threshold")
                dialog.show(supportFragmentManager, "Select eye threshold")
                return true
            }
            R.id.delayBeforeSignal -> {
                val dialog = EyeThresholdDialog(
                    {
                        delayBeforeDetectionLock.withLock {
                            delayBeforeDetectionRising = it
                        }
                    },
                    R.id.editEyeThreshold, "Select rising detection delay")
                dialog.show(supportFragmentManager, "Select rising detection delay")
                return true
            }
            R.id.delayBeforeSignalFalling -> {
                val dialog = EyeThresholdDialog(
                {
                    delayBeforeDetectionLock.withLock {
                        delayBeforeDetectionFalling = it
                    }
                },
                R.id.editEyeThreshold, "Select falling detection delay")
            dialog.show(supportFragmentManager, "Select falling detection delay")
                return true
        }
    }
    return super.onOptionsItemSelected(item)
    }
}

private fun Float.format(digits: Int): String = "%.${digits}f".format(this)

private operator fun Point.plus(other: Point?): Point? {
    if (other != null) {
        return Point(this.x + other.x, this.y + other.y)
    } else {
        return null
    }
}

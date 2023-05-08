package com.younglords.imagerecognitionapp

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.younglords.imagerecognitionapp.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var selectImageBtn: Button
    private lateinit var makePredictionBtn: Button
    private lateinit var imageView: ImageView
    private lateinit var textView: TextView
    private lateinit var bitMap: Bitmap


    private fun checkAndGrantPermissions() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        } else {
            Toast.makeText(this, "Camera permission is granted", Toast.LENGTH_SHORT).show()
        }
    }


    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera permission is granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Camera permission is denied", Toast.LENGTH_SHORT).show()
            }
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectImageBtn = findViewById(R.id.select_image_btn)
        makePredictionBtn = findViewById(R.id.make_prediction_btn)
        imageView = findViewById(R.id.image_view)
        textView = findViewById(R.id.text_view)

        checkAndGrantPermissions()

        selectImageBtn.setOnClickListener {
            selectImage()
        }
        makePredictionBtn.setOnClickListener {
            makePrediction()
        }

    }

    private fun makePrediction() {
        val labels =
            application.assets.open("labels.txt").bufferedReader().use { it.readText() }.split("\n")

        val resized = Bitmap.createScaledBitmap(bitMap, 224, 224, true)
        val model = MobilenetV110224Quant.newInstance(this)

        val tBuffer = TensorImage.fromBitmap(resized)
        val byteBuffer = tBuffer.buffer

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
        inputFeature0.loadBuffer(byteBuffer)


        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val max = getMax(outputFeature0.floatArray)


        textView.text = labels[max]
        model.close()
    }


    private fun selectImage() {

        val items = arrayOf<CharSequence>(
            "Take Photo", "Choose from Library", "Cancel"
        )

        val dialogBox = AlertDialog.Builder(this)
        dialogBox.setItems(items) { dialog, item ->
            when {
                items[item].contentEquals("Take Photo") -> {
                    openCamera()
                }
                items[item].contentEquals("Choose from Library") -> {
                    chooseImageFromGallery()
                }
                items[item].contentEquals("Cancel") -> {
                    dialog.dismiss()
                }
            }
        }
        dialogBox.setCancelable(true)
        dialogBox.show()
    }

    private fun chooseImageFromGallery() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"
        intent.putExtra("crop", "true")
        intent.putExtra("scale", true)
        intent.putExtra("return-data", true)
        startActivityForResult(intent, 250)
    }

    private fun openCamera() {
        val camera = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(camera, 200)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 250) {
            imageView.setImageURI(data?.data)
            val uri: Uri? = data?.data
            bitMap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
        } else if (requestCode == 200 && resultCode == Activity.RESULT_OK) {
            bitMap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(bitMap)
        }

    }

    private fun getMax(arr: FloatArray): Int {
        var ind = 0
        var min = 0.0f
        for (i in 0..1000) {
            if (arr[i] > min) {
                min = arr[i]
                ind = i
            }
        }
        return ind
    }
}
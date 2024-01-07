package id.kukly.frankenstein.ui.detection

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import androidx.navigation.findNavController
import by.kirich1409.viewbindingdelegate.viewBinding
import dagger.hilt.android.AndroidEntryPoint
import id.kukly.frankenstein.Api.Resource
import id.kukly.frankenstein.R
import id.kukly.frankenstein.databinding.FragmentDetectionBinding
import kotlinx.coroutines.flow.collectLatest
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import java.io.ByteArrayOutputStream

@AndroidEntryPoint
class DetectionFragment : Fragment(R.layout.fragment_detection) {

    private val CAMERA_PERMISSION_CODE = 100
    private val binding by viewBinding(FragmentDetectionBinding::bind)
    private var step = 1
    private val detectionViewModel : DetectionViewModel by activityViewModels()
//    Log.i("Binding", "predictedClass1: $predictedClass1")

    private var conditionDiagnosa = false
//    private val modelRepository = ModelRepository(RetrofitInstance.apiService)




    private val launcherIntentCamera = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (it.resultCode == Activity.RESULT_OK) {
            val imageBitmap = it.data?.extras?.get("data") as Bitmap
            initializeBitmapImageView(imageBitmap)
        }
    }

    private val launcherIntentGallery = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val imageUri = result.data?.data as Uri
            initializeUriImageView(imageUri)
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)


        initButtonCamera()
        initButtonGallery()

        // Move to Frame
        toHome()
        toAbout()
        toProfile()


        // Check Detection Button
        binding.checkDetection.setOnClickListener {
           performDetection()
        }
    }

    private fun performDetection() = with(binding) {
        Log.i("My App", "Preparing to send prediction request")

        val imageAtas = getImagePart(binding.imageSisiAtas,"atas")
        val imageBawah = getImagePart(binding.imageSisiBawah,"bawah")
        val imageSamping = getImagePart(binding.imageSisiSamping,"samping")


        //kata ka rafi kemungkinan dari send gambar yang dikirimnya itu filenya sama makanya predictnya salah
        println(binding.imageSisiAtas)
        println(binding.imageSisiBawah)
        println(binding.imageSisiSamping)

        Log.d("try fun atassss" , "imgAtas ${binding.imageSisiAtas}")
        Log.d("try fun bawahhh" , "imgAtas ${binding.imageSisiBawah}")
        Log.d("try fun sampinggg" , "imgAtas ${binding.imageSisiSamping}")

        // Show progress bar
        progressBar.isVisible = true

        detectionViewModel.predictNorab(imageAtas,imageBawah,imageSamping)
        observePredicrNorab()
    }

    private fun observePredicrNorab() = with(binding){
        Log.d("observePredicrNorab", "masuk sini")
        lifecycleScope.launchWhenStarted {
            detectionViewModel.predictNorab.collectLatest {
                it?.let { response ->
                    when(response){
                        is Resource.Error ->{
                            Log.d("observePredicrNorab", "Resource.Error")
                            progressBar.isVisible = false
                        }
                        is Resource.Loading ->{
                            Log.d("observePredicrNorab", "Resource.Loading")

                            progressBar.isVisible = true
                        }
                        is Resource.Success ->{
                            Log.d("observePredicrNorab", "Resource.Succses")
                            progressBar.isVisible = false

                            // Kodingan tambahan
                            val predictionLowerCase = response.data?.finalPrediction?.toLowerCase()
                            updateTvHasilPredictBackground(predictionLowerCase == "normal")
                            // Selesai Hapus jika ada error

                            binding.tvHasilPredict.text = response.data?.finalPrediction.toString()

                        }
                    }
                }
            }
        }
    }

    // Kodingan Tambahan
    private fun updateTvHasilPredictBackground(isNormal: Boolean) {
        val widthDp = resources.getDimensionPixelSize(R.dimen.tv_hasil_predict_width)
        val heightDp = resources.getDimensionPixelSize(R.dimen.tv_hasil_predict_height)
        if (isNormal) {
            binding.tvHasilPredict.apply {
                setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.colorNormal))
                setTextColor(ContextCompat.getColor(requireContext(), android.R.color.white)) // Set text color to white
                layoutParams.width = widthDp
                layoutParams.height = heightDp
            }
        } else {
            binding.tvHasilPredict.apply {
                setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.colorAbnormal))
                setTextColor(ContextCompat.getColor(requireContext(), android.R.color.white)) // Set text color to white
                layoutParams.width = widthDp
                layoutParams.height = heightDp
            }
        }
        // Untuk memastikan perubahan layout diterapkan
        binding.tvHasilPredict.requestLayout()
    }

    private fun getImagePart(imageView: View, posisi:String): MultipartBody.Part {
        val bitmap = getBitmapFromImageView(imageView)
        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream)
        val requestBody =
            RequestBody.create("image/*".toMediaTypeOrNull(), byteArrayOutputStream.toByteArray())
        return MultipartBody.Part.createFormData(posisi, "image.jpg", requestBody)

    }

    private fun getBitmapFromImageView(imageView: View): Bitmap {
//        return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        val drawable = (imageView as ImageView).drawable
        val bitmap = Bitmap.createBitmap(
            drawable.intrinsicWidth,
            drawable.intrinsicHeight,
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, canvas.width, canvas.height)
        drawable.draw(canvas)
        return bitmap
    }


    //function to Home frame
    private fun toHome() = with(binding){
        iconHome.setOnClickListener{
            binding.iconHome.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_detectionFragment_to_mainFragment)
        }
    }
    //function to About Frame
    private fun toAbout() = with(binding){
        iconAbout.setOnClickListener{
            binding.iconAbout.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_detectionFragment_to_aboutFragment)
        }
    }

    private fun toProfile() = with(binding){
        iconUser.setOnClickListener{
            binding.iconUser.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_detectionFragment_to_profileFragment)
        }
    }

    private fun initButtonGallery() = with(binding){
        iconGaleriAtas.setOnClickListener {
            step = 1
            startGallery()
        }
        iconGaleriBawah.setOnClickListener {
            step = 2
            startGallery()
        }
        iconGaleriSamping.setOnClickListener {
            step = 3
            startGallery()
        }
    }

    private fun initButtonCamera()= with(binding){
        iconKameraAtas.setOnClickListener {
            step = 1
            checkCameraPermission()
        }
        iconKameraBawah.setOnClickListener {
            step = 2
            checkCameraPermission()}
        iconKameraSamping.setOnClickListener {
            step = 3
            checkCameraPermission()}
    }

    private fun initializeBitmapImageView(imageBitmap:Bitmap ) = with(binding){
        when (step){
            1 -> {imageSisiAtas.setImageBitmap(imageBitmap)}
            2 -> {imageSisiBawah.setImageBitmap(imageBitmap)}
            3 -> {imageSisiSamping.setImageBitmap(imageBitmap)}
        }
    }

    private fun initializeUriImageView(imageUri : Uri) = with(binding){
        when(step){
            1 -> {imageSisiAtas.setImageURI(imageUri)}
            2 -> {imageSisiBawah.setImageURI(imageUri)}
            3 -> {imageSisiSamping.setImageURI(imageUri)}

        }
    }


    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // Jika izin tidak diberikan, minta izin
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        } else {
            // Izin telah diberikan, lanjutkan dengan aksi yang diperlukan
            launchCamera()
        }
    }

    private fun launchCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        launcherIntentCamera.launch(intent)

    }

    private fun startGallery(){
        val intent = Intent()
        intent.action = Intent.ACTION_GET_CONTENT
        intent.type = "image/*"
        val chooser = Intent.createChooser(intent, "Choose a Picture")
        launcherIntentGallery.launch(chooser)
    }
}
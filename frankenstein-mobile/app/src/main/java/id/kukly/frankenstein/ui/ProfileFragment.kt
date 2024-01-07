package id.kukly.frankenstein.ui

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.findNavController
import by.kirich1409.viewbindingdelegate.viewBinding
import com.bumptech.glide.Glide
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener
import com.google.firebase.storage.FirebaseStorage
import com.google.firebase.storage.StorageReference
import id.kukly.frankenstein.R
import id.kukly.frankenstein.dataUser.UserProfile
import id.kukly.frankenstein.databinding.FragmentProfileBinding
import java.io.InputStream

class ProfileFragment : Fragment(R.layout.fragment_profile) {
    private val binding by viewBinding(FragmentProfileBinding::bind)
    private val database = FirebaseDatabase.getInstance()
    private val auth: FirebaseAuth = FirebaseAuth.getInstance()
    private val userRef = database.getReference("users").child(auth.currentUser?.uid ?: "")
    private val storage = FirebaseStorage.getInstance()
    private val storageRef: StorageReference = storage.reference
    private var selectedImageUri: Uri? = null  // Tambahkan variabel global untuk menyimpan URI gambar yang dipilih
    val imageRef: StorageReference =
        storageRef.child("user_images/${auth.currentUser?.uid ?: ""}/profile_picture.jpg")




    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        toHome()
        toAbout()

        initIconGallery()
        initSaveButton()
        initFirebaseDataListener()
    }

    private fun toHome() = with(binding){
        iconHome.setOnClickListener{
            binding.iconHome.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_profileFragment_to_mainFragment)
        }
    }

    private fun toAbout() = with(binding){
        iconAbout.setOnClickListener{
            binding.iconAbout.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_profileFragment_to_aboutFragment)
        }
    }

//    Kode Icon to Choose pict in galery
    private fun initIconGallery() = with(binding) {
        userPhoto.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, GALLERY_REQUEST_CODE)
        }
    }

    private fun uploadImageToFirebaseStorage(imageUri: Uri) {

        // Ambil InputStream dari gambar yang dipilih
        val inputStream: InputStream? = requireContext().contentResolver.openInputStream(imageUri)

        // Upload gambar ke storage
        if (inputStream != null) {
            storageRef.child("user_images/${auth.currentUser?.uid ?: ""}/profile_picture.jpg")
                .putStream(inputStream)
                .addOnSuccessListener {
                    // Handle jika upload berhasil
                    Toast.makeText(
                        requireContext(),
                        "Image uploaded successfully!",
                        Toast.LENGTH_SHORT
                    ).show()

                    // Ambil URL gambar yang diunggah
                    getImageDownloadUrl(storageRef.child("user_images/${auth.currentUser?.uid ?: ""}/profile_picture.jpg"))
                }
                .addOnFailureListener { e ->
                    // Handle jika upload gagal
                    Toast.makeText(
                        requireContext(),
                        "Failed to upload image: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
        } else {
            Toast.makeText(requireContext(), "Failed to open image file", Toast.LENGTH_SHORT).show()
        }
    }


    private fun getImageDownloadUrl(imageRef: StorageReference) {
        imageRef.downloadUrl
            .addOnSuccessListener { uri ->
                // Tambahkan log atau toast untuk memastikan URL berhasil diperoleh
                Log.d("Firebase", "Image download URL: $uri")

                // Simpan URL gambar ke database
                saveDataToFirebase(uri.toString())

                // Setelah mendapatkan URL, perbarui UI
                updateUI(UserProfile(binding.inputNama.text.toString(), binding.inputEmail.text.toString(), binding.inputPhone.text.toString(), uri.toString()))
            }
            .addOnFailureListener { e ->
                Toast.makeText(
                    requireContext(),
                    "Failed to get image download URL: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
            }
    }


    companion object {
        private const val GALLERY_REQUEST_CODE = 123
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == GALLERY_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            data?.data?.let { selectedImageUri ->
                // Lakukan sesuatu dengan gambar yang dipilih, misalnya tampilkan di ImageView
                binding.userPhoto.setImageURI(selectedImageUri)
                this.selectedImageUri = selectedImageUri  // Simpan URI gambar yang dipilih
            }
        }
    }

    //
    private fun saveDataToFirebase(imageUrl: String? = null) {
        val nama = binding.inputNama.text.toString()
        val email = binding.inputEmail.text.toString()
        val phone = binding.inputPhone.text.toString()
        //

        val userProfile = UserProfile(nama, email, phone, imageUrl ?: "")

        // Dapatkan referensi database
        val database = FirebaseDatabase.getInstance()
        val userRef = database.getReference("users").child(auth.currentUser?.uid ?: "")

        // Simpan data ke Firebase
        userRef.setValue(userProfile)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    Toast.makeText(requireContext(), "Data saved to Firebase!", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(requireContext(), "Failed to save data: ${task.exception?.message}", Toast.LENGTH_SHORT).show()
                }
            }
    }

    //panggil save data to firebase
    private fun initSaveButton() {
        binding.buttonSave.setOnClickListener {

            // Periksa apakah ada gambar yang dipilih
            if (selectedImageUri != null) {
                // Jika ada gambar yang dipilih, upload gambar ke Firebase Storage
                uploadImageToFirebaseStorage(selectedImageUri!!)
            }

            saveDataToFirebase()
//            getImageDownloadUrl(imageRef)
        }
    }

    //Monitor perubahan data
    private fun initFirebaseDataListener() {
        userRef.addValueEventListener(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                if (snapshot.exists()) {
                    val userProfile = snapshot.getValue(UserProfile::class.java)
                    updateUI(userProfile)
                }
            }

            override fun onCancelled(error: DatabaseError) {
                Toast.makeText(requireContext(), "Failed to read data: ${error.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    //Update UI
    private fun updateUI(userProfile: UserProfile?) {
        userProfile?.let {
            binding.inputNama.setText(it.nama)
            binding.inputEmail.setText(it.email)
            binding.inputPhone.setText(it.phone)

            loadProfileImage()
        }
    }


    private fun loadProfileImage() {
        // Ambil referensi ke gambar profil di Firebase Storage
        val profileImageRef = storageRef.child("user_images/${auth.currentUser?.uid ?: ""}/profile_picture.jpg")

        // Muat gambar dari Firebase Storage
        profileImageRef.downloadUrl
            .addOnSuccessListener { uri ->
                // Tampilkan gambar di ImageView
                Glide.with(requireContext())
                    .load(uri)
                    .into(binding.userPhoto)
            }
            .addOnFailureListener { e ->
                Log.e("Firebase", "Failed to load profile image: ${e.message}")
            }
    }



}
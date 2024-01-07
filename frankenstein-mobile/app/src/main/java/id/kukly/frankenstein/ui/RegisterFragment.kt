package id.kukly.frankenstein.ui

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import by.kirich1409.viewbindingdelegate.viewBinding
import com.google.firebase.auth.FirebaseAuth
import id.kukly.frankenstein.R
import id.kukly.frankenstein.databinding.FragmentRegisterBinding


class RegisterFragment : Fragment(R.layout.fragment_register) {

    private lateinit var auth : FirebaseAuth
    private val binding by viewBinding(FragmentRegisterBinding::bind)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        auth = FirebaseAuth.getInstance()

        binding.tvLogin.setOnClickListener {
            binding.tvLogin.setTextColor(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            findNavController().navigate(R.id.action_registerFragment_to_loginFragment2)
        }

        binding.buttonRegister.setOnClickListener {
            val email = binding.inputEmail.text.toString()
            val password = binding.inputPassword.text.toString()
            val confirmPassword = binding.inputConfirmPassword.text.toString()

            if (email.isNotEmpty() && password.isNotEmpty() && confirmPassword.isNotEmpty()) {
                if (password == confirmPassword) {
                    auth.createUserWithEmailAndPassword(email, password).addOnCompleteListener {
                        if (it.isSuccessful) {
                            Toast.makeText(requireContext(), "Registration successful!", Toast.LENGTH_SHORT).show()
                            findNavController().navigate(R.id.action_registerFragment_to_mainFragment)
                        } else {
                            Toast.makeText(requireContext(), it.exception?.message, Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    Toast.makeText(requireContext(), "Password is not matching", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(requireContext(), "Empty Fields Are not Allowed!", Toast.LENGTH_SHORT).show()
            }
        }
    }

//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        binding = FragmentRegisterBinding.inflate(layoutInflater)
//        setContentView(binding.root)
//
//        auth = FirebaseAuth.getInstance()
//
//        binding.tvLogin.setOnClickListener {
//            val intent = Intent(this, FragmentLoginBinding::class.java)
//            startActivity(intent)
//        }
//
//        binding.buttonRegister.setOnClickListener{
//            val email = binding.inputEmail.text.toString()
//            val password = binding.inputPassword.text.toString()
//            val confirmPassword = binding.inputConfirmPassword.text.toString()
//
//            if(email.isNotEmpty() && password.isNotEmpty() && confirmPassword.isNotEmpty()){
//                if (password == confirmPassword){
//                    auth.createUserWithEmailAndPassword(email, password).addOnCompleteListener{
//                        if(it.isSuccessful){
//                            val intent = Intent(this, FragmentRegisterBinding::class.java)
//                            startActivity(intent)
//                        }else{
//                            Toast.makeText(this, it.exception.toString(), Toast.LENGTH_SHORT).show()
//                        }
//                    }
//                }else{
//                    Toast.makeText(this, "Password is not matching", Toast.LENGTH_SHORT).show()
//                }
//            }else{
//                Toast.makeText(this, "Empty Fields Are not Allowed !", Toast.LENGTH_SHORT).show()
//            }
//        }
//    }






//    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
//        super.onViewCreated(view, savedInstanceState)
//        toLogin()
//        registerButton()
//
//        auth = FirebaseAuth.getInstance()
//
//    }
//
//
//
//
//    private fun toLogin() = with(binding){
//        tvLogin.setOnClickListener{
//            it.findNavController().navigate(R.id.action_registerFragment_to_loginFragment2)
//        }
//    }
//
//    private fun registerButton() = with(binding){
//        buttonRegister.setOnClickListener{
//            val email = binding.inputEmail.text.toString()
//            val password = binding.inputPassword.text.toString()
//            val confirmPassword = binding.inputConfirmPassword.text.toString()
//
//            if (email.isEmpty()){
//                binding.inputEmail.error = "Email Harus di Isi"
//            }
//
//            it.findNavController().navigate(R.id.action_registerFragment_to_mainFragment)
//        }
//    }
}
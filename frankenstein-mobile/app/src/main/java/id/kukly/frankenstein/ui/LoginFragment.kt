package id.kukly.frankenstein.ui

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.findNavController
import androidx.navigation.fragment.findNavController
import by.kirich1409.viewbindingdelegate.viewBinding
import com.google.firebase.auth.FirebaseAuth
import id.kukly.frankenstein.R
import id.kukly.frankenstein.databinding.FragmentLoginBinding


class LoginFragment : Fragment(R.layout.fragment_login) {

    private val binding by viewBinding(FragmentLoginBinding::bind)
    private lateinit var auth : FirebaseAuth

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        auth = FirebaseAuth.getInstance()
        toRegister()
        toMain()
    }

    private fun toRegister() = with(binding) {
        tvRegister.setOnClickListener {
            binding.tvRegister.setTextColor(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_loginFragment_to_registerFragment)
        }
    }

    private fun toMain() = with(binding) {
        buttonLogin.setOnClickListener {
            val email = inputEmail.text.toString()
            val password = inputPassword.text.toString()

            if (email.isNotEmpty() && password.isNotEmpty()) {
                auth.signInWithEmailAndPassword(email, password).addOnCompleteListener { task ->
                    if (task.isSuccessful) {
                        Toast.makeText(requireContext(), "Login successful!", Toast.LENGTH_SHORT).show()
                        findNavController().navigate(R.id.action_loginFragment_to_mainFragment)
                    } else {
                        Toast.makeText(requireContext(), task.exception?.message, Toast.LENGTH_SHORT).show()
                    }
                }
            } else {
                Toast.makeText(requireContext(), "Empty Fields Are not Allowed!", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
package id.kukly.frankenstein.ui

import android.os.Bundle
import android.view.View
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.findNavController
import by.kirich1409.viewbindingdelegate.viewBinding
import id.kukly.frankenstein.R
import id.kukly.frankenstein.databinding.FragmentAboutBinding


class AboutFragment : Fragment(R.layout.fragment_about) {
    private val binding by viewBinding(FragmentAboutBinding::bind)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        toHome()
        toProfile()
    }

    private fun toHome() = with(binding){
        iconHome.setOnClickListener{
            binding.iconHome.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_aboutFragment_to_mainFragment)
        }
    }

    private fun toProfile() = with(binding){
        iconUser.setOnClickListener{
            binding.iconUser.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_aboutFragment_to_profileFragment)
        }
    }

}
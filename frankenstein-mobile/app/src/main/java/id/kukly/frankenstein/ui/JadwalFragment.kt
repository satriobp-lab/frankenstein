package id.kukly.frankenstein.ui

import android.os.Bundle
import android.view.View
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.findNavController
import by.kirich1409.viewbindingdelegate.viewBinding
import id.kukly.frankenstein.R
import id.kukly.frankenstein.databinding.FragmentJadwalBinding


class JadwalFragment : Fragment(R.layout.fragment_jadwal) {
    private val binding by viewBinding(FragmentJadwalBinding::bind)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        toHome()
        toAbout()
        toProfile()
    }

    private fun toHome() = with(binding){
        iconHome.setOnClickListener{
            binding.iconHome.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_jadwalFragment_to_mainFragment)
        }
    }

    private fun toAbout() = with(binding){
        iconAbout.setOnClickListener{
            binding.iconAbout.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_jadwalFragment_to_aboutFragment)
        }
    }

    private fun toProfile() = with(binding){
        iconUser.setOnClickListener{
            binding.iconUser.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_jadwalFragment_to_profileFragment)
        }
    }
}
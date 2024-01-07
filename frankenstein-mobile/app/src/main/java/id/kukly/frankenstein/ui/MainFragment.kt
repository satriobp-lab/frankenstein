package id.kukly.frankenstein.ui

import android.os.Bundle
import android.view.View
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.findNavController
import by.kirich1409.viewbindingdelegate.viewBinding
import id.kukly.frankenstein.R
import id.kukly.frankenstein.databinding.FragmentMainBinding


class MainFragment : Fragment(R.layout.fragment_main) {
    private val binding by viewBinding(FragmentMainBinding::bind)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        toDetection()
        toSchedule()
        toAbout()
        toProfile()
    }

    private fun toDetection() = with(binding){
        buttonDeteksi.setOnClickListener{
            it.findNavController().navigate(R.id.action_mainFragment_to_detectionFragment)
        }
    }

    private fun toSchedule() = with(binding){
        buttonJadwal.setOnClickListener{
            it.findNavController().navigate(R.id.action_mainFragment_to_jadwalFragment)
        }
    }

    private fun toAbout() = with(binding){
        iconAbout.setOnClickListener{
            binding.iconAbout.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_mainFragment_to_aboutFragment)
        }
    }

    private fun toProfile() = with(binding){
        iconUser.setOnClickListener{
            binding.iconUser.setColorFilter(ContextCompat.getColor(requireContext(), R.color.icon_pressed))
            it.findNavController().navigate(R.id.action_mainFragment_to_profileFragment)
        }
    }
}
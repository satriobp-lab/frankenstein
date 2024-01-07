package id.kukly.frankenstein.ui.detection

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import id.kukly.frankenstein.Api.Resource
import id.kukly.frankenstein.Api.repository.NorabRepository
import id.kukly.frankenstein.Api.response.NorabResponse
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import okhttp3.MultipartBody
import javax.inject.Inject

@HiltViewModel
class DetectionViewModel @Inject constructor(
    private val norabRepository: NorabRepository) : ViewModel(){

    private val _predicNorab = MutableStateFlow<Resource<NorabResponse>?>(null)
    val predictNorab  : MutableStateFlow<Resource<NorabResponse>?> = _predicNorab


    fun predictNorab (imageAtas : MultipartBody.Part, imageBawah : MultipartBody.Part, imageSamping : MultipartBody.Part) = viewModelScope.launch {
//        _predicNorab.value = Resource.Loading()
        Log.d(" fun vm predictNorab", "masuk sini")
        norabRepository.uploadImage(imageAtas,imageBawah,imageSamping).collectLatest {
            Log.d("norabRepositoryfun vm predictNorab", "masuk sini")
            _predicNorab.value=it
        }

    }
}
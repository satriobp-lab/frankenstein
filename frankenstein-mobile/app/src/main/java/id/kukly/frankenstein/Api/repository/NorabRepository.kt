package id.kukly.frankenstein.Api.repository

import id.kukly.frankenstein.Api.Resource
import id.kukly.frankenstein.Api.response.NorabResponse
import kotlinx.coroutines.flow.Flow
import okhttp3.MultipartBody

interface NorabRepository {

    suspend fun uploadImage(imageFileAtas: MultipartBody.Part,
                            imageFileBawah: MultipartBody.Part,
                            imageFileSamping: MultipartBody.Part,) : Flow<Resource<NorabResponse>>

}
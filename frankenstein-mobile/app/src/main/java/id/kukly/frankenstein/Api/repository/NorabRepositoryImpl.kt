package id.kukly.frankenstein.Api.repository

import android.util.Log
import id.kukly.frankenstein.Api.Resource
import id.kukly.frankenstein.Api.network.ApiService
import id.kukly.frankenstein.Api.response.NorabResponse
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import okhttp3.MultipartBody
import javax.inject.Inject


class NorabRepositoryImpl @Inject constructor(
    private val apiService: ApiService
) : NorabRepository{
    override suspend fun uploadImage(
        imageFileAtas: MultipartBody.Part,
        imageFileBawah: MultipartBody.Part,
        imageFileSamping: MultipartBody.Part
    ): Flow<Resource<NorabResponse>>  = flow{
        try {
            Log.d("try fun uploadImage repo" , "imgAtas $imageFileAtas")
            val response = apiService.sendPredictionResult(imageFileAtas,imageFileBawah,imageFileSamping)
            emit(Resource.Success(response))
        } catch (e : Exception){
            e.printStackTrace()
            emit(Resource.Error("Unknown error occurred"))
        }
    }
}
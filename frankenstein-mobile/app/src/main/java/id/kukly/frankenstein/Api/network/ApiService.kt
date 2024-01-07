package id.kukly.frankenstein.Api.network
import id.kukly.frankenstein.Api.response.NorabResponse
import okhttp3.MultipartBody
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {
        @Multipart
        @POST("getpredict")
        suspend fun sendPredictionResult(
                @Part atas: MultipartBody.Part,
                @Part bawah: MultipartBody.Part,
                @Part samping: MultipartBody.Part
        ): NorabResponse
}

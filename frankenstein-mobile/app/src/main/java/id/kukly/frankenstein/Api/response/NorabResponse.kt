package id.kukly.frankenstein.Api.response

import com.google.gson.annotations.SerializedName

data class NorabResponse(

	@field:SerializedName("predicted_class_1")
	val predictedClass1: String? = null,

	@field:SerializedName("predicted_class_2")
	val predictedClass2: String? = null,

	@field:SerializedName("predicted_class_3")
	val predictedClass3: String? = null,

	@field:SerializedName("final_prediction")
	val finalPrediction: String? = null
)

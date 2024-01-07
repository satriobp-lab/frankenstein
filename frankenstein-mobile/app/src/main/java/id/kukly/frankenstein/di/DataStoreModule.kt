package id.kukly.frankenstein.di

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import id.kukly.frankenstein.Api.network.ApiConfig
import id.kukly.frankenstein.Api.network.ApiService
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
class DataStoreModule {

    @Singleton
    @Provides
    fun provideUploadService(): ApiService = ApiConfig.getUploadService()

}
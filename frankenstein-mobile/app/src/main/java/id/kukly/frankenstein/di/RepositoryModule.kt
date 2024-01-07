package id.kukly.frankenstein.di

import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import id.kukly.frankenstein.Api.repository.NorabRepository
import id.kukly.frankenstein.Api.repository.NorabRepositoryImpl

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    abstract fun bindsNorabRepository(norabRepository: NorabRepositoryImpl) : NorabRepository
}
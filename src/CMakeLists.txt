add_subdirectory(internal_tools)
add_subdirectory(ai_models)

if(TK_ENABLE_GPU)
  add_subdirectory(gpu)
endif()

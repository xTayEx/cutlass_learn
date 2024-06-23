add_rules("mode.debug", "mode.release")

-- add_requires("thrust")
target("cutlass_learn")
    set_kind("binary")
    add_files("src/main.cu")

    -- generate SASS code for SM architecture of current host
    add_cugencodes("native")

    -- generate PTX code for the virtual architecture to guarantee compatibility
    add_cugencodes("compute_89")

    -- add_packages("thrust")
    
    add_includedirs("3rdparty/cutlass/include")
    add_includedirs("3rdparty/cutlass/tools/util/include")

    -- -- generate SASS code for each SM architecture
    -- add_cugencodes("sm_35", "sm_37", "sm_50", "sm_52", "sm_60", "sm_61", "sm_70", "sm_75")

    -- -- generate PTX code from the highest SM architecture to guarantee forward-compatibility
    -- add_cugencodes("compute_75")

target("ref")
    set_kind("binary")
    add_files("src/ref.cu")

    -- generate SASS code for SM architecture of current host
    add_cugencodes("native")

    -- generate PTX code for the virtual architecture to guarantee compatibility
    add_cugencodes("compute_89")

    -- add_packages("thrust")
    
    add_includedirs("3rdparty/cutlass/include")
    add_includedirs("3rdparty/cutlass/tools/util/include")

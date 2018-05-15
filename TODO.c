dataset_generation
    square_png_files -> filtered_square_png_files -> dataset.h5

batch_generation
    h5file_name -> h5dataset -> shuffled_h5dataset => batches

train
    parameter initilization(from cmd)
    model definitions
    loop(train)
    save model & result image

layers

test
    interactive_test
    evaluate
ui

utilities
    timer


--------
1. generate dataset
2. generate batch
3. test and evaluate
4. training
5. fine-tuning

--------
write unit-test using small dataset.
write LOOK & FEEL test using big dataset.
must measure running time.(profiling)

--------
(*) write unit test codes for modules.
(*) batch_gen | training are must be parallelized.
(*) try profiler
(*) try RxPy

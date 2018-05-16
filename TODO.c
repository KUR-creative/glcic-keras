(imgedit): get square files for feeding
    original h != w image
    -> cutting -> filtering -> square_png_files

dataset_generation: get h5 file dataset for training
    square_png_files -> filtered_square_png_files -> dataset.h5
    h5 file save & load

batch_generation: get batch from h5 dataset for training
    h5file_name 
    -> h5dataset 
    -> shuffled_h5dataset 
    -> runtime augmentation
    => batches

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

(*) use tqdm, parallelization.



ex: img_128x128_10k.h5
includes: (로드하면 바로 쓸 수 있게 해둠)
    RGB, regularized, square images
    regularized mean_pixel_value of dataset

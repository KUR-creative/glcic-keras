import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset_name',
    help='dataset name. generated hdf5 file required.',
    required=True)  

parser.add_argument('-c', '--is_color_dataset',
    help='dataset is color images? yes:-c no:(no -c)', 
    type=bool, default=False, nargs='?') 
    # False = (no -C) = grayscale, None = -C = rgb.

parser.add_argument('-b', '--batch_size',
    help='batch size. 2^x recommended.', 
    type=int, required=True)  
parser.add_argument('-i', '--img_size',
    help='image size. 2^x recommended.',
    type=int, required=True)  

parser.add_argument('-l', '--ld_crop_size_proportion',
    help='proportion of local discrimnator,default:0.5 0 < L < 1',
    type=float, default=0.5)  

parser.add_argument('-M', '--random_hole_max_proportion',
    help='maximum size proportion of random hole in local crop(default:1.0). 0 < M <= 1',
    type=float, default=1.0) 
parser.add_argument('-m', '--random_hole_min_proportion',
    help='minimum size proportion of random hole in local crop(default:0.5). 0 < m <= 1',
    type=float, default=0.5)  

parser.add_argument('-a', '--alpha',
    help='joint model hyperparameter(default:0.0004). 0 < a < 1',
    type=float, default=0.0004)  
parser.add_argument('-lr', '--learning_rate',
    help=' learning rate of Dmodel(default:0.01). 0 < lr < 1',
    type=float, default=0.01)  

parser.add_argument('-ne', '--num_epoch',
    help='number of epochs. 0 <= ne',
    type=int, required=True)  
parser.add_argument('-tc', '--tc_ratio',
    help='portion of epochs C model(default:0.18). 0 < tc < 1',
    type=float, default=0.18)  
parser.add_argument('-td', '--td_ratio',
    help='portion of epochs D model(default:0.02). 0 < td < 1',
    type=float, default=0.02)  

parser.add_argument('-dr', '--learned_data_ratio',
    help='ratio of data to use from full dataset',
    type=float, default=1.0)
parser.add_argument('-si', '--save_interval',
    help='save interval of epochs. 1 <= si < num_epoch',
    type=int, required=True)  

parser.add_argument('-me', '--mailing_enabled',
    help='is mailing enabled? yes:-me no:(no -me)', 
    type=bool, default=False, nargs='?') 
    # False = (no -me) = disabled, None = -me = enabled.

parser.add_argument('-C', '--c_model_path',
    help='compl_model path to load', type=str)  
parser.add_argument('-D', '--d_model_path',
    help='discrim_model path to load', type=str)  
parser.add_argument('-ce', '--current_epoch',
    help='current epoch of loaded model. 0 <= ce',
    type=int, default=0)

args = parser.parse_args()

if not (0.0 < args.ld_crop_size_proportion
        and   args.ld_crop_size_proportion < 1.0):
    parser.error('[require] 0.0 < proportion of local discrimnator < 1.0')

if not (0.0 < args.random_hole_min_proportion 
        and   args.random_hole_min_proportion <= 1.0):
    parser.error('[require] 0.0 < minimum proportion of hole size <= 1.0')
if not (0.0 < args.random_hole_max_proportion 
        and   args.random_hole_max_proportion <= 1.0):
    parser.error('[require] 0.0 < maximum proportion of hole size <= 1.0')

if not (0.0 < args.alpha and args.alpha < 1.0):
    parser.error('[require] 0.0 < alpha of joint model < 1.0')
if not (0.0 < args.learning_rate): 
    parser.error('[require] 0.0 < learning rate of Discrimnator model')
if not (1 <= args.save_interval and args.save_interval < args.num_epoch): 
    parser.error('[require] 1 <= save interval of epochs. < num_epoch')

if not (0.0 < args.learned_data_ratio and args.learned_data_ratio <= 1.0):
    parser.error('[require] 0.0 < learned_data_ratio <= 1.0')

if not (0 <= args.num_epoch): 
    parser.error('[require] 0 <= number of epochs')
if not (0.0 < args.tc_ratio and args.tc_ratio < 1.0):
    parser.error('[require] 0.0 < alpha of joint model < 1.0')
if not (0.0 < args.td_ratio and args.td_ratio < 1.0):
    parser.error('[require] 0.0 < alpha of joint model < 1.0')

# relations between some variables
if not (args.random_hole_min_proportion < args.random_hole_max_proportion):
    parser.error('[require] min hole size < max hole size ')

if not (args.tc_ratio + args.td_ratio < 1.0):
    parser.error('[require] tc_ratio + td_ratio < 1.0')

if not (0 <= args.current_epoch): 
    parser.error('[require] 0 <= current epoch of loaded model.')

if ((args.c_model_path == None) ^ (args.d_model_path == None)):
    parser.error('[require] Both C and D models are required. ')


import sys
if __name__ == "__main__":
    import model
    model.set_global_consts(args.batch_size, args.img_size,
                            True if args.is_color_dataset == False else False,
                            args.ld_crop_size_proportion,
                            args.random_hole_min_proportion, args.random_hole_max_proportion,
                            args.alpha, args.learning_rate)
    tc = int(args.num_epoch * args.tc_ratio)
    td = int(args.num_epoch * args.td_ratio)  
    t_joint = args.num_epoch - tc - td
    print(' '.join(sys.argv),'\n')
    print(' ==============SUMMARY==============\n',
          '    dataset name = %s \n' % args.dataset_name,
          '    dataset type = %s \n' % ('grayscale' if args.is_color_dataset == False else 'rgb'), 
          '      batch size = %d \n' % model.BATCH_SIZE, 
          '      image size = %d \n' % model.IMG_SIZE,
          '-----------------------------------\n',
          '     image shape = %s \n' % str(model.IMG_SHAPE),
          '   ld crop shape = %s \n' % str(model.LD_CROP_SHAPE),
          '      mask shape = %s \n' % str(model.MASK_SHAPE),
          '-----------------------------------\n',
          '   ld crop ratio = %f \n' % args.ld_crop_size_proportion,
          '  hole min ratio = %f \n' % args.random_hole_min_proportion,
          '  hole max ratio = %f \n' % args.random_hole_max_proportion,
          '    ld crop size = %d \n' % model.LD_CROP_SIZE,
          '   hole min size = %d \n' % model.HOLE_MIN_LEN,
          '   hole max size = %d \n' % model.HOLE_MAX_LEN,
          '-----------------------------------\n',
          'hyperparam alpha = %f \n' % model.ALPHA, 
          '   hyperparam lr = %f \n' % model.D_MODEL_LR,
          '-----------------------------------\n',
          '        Tc ratio = %f \n' % args.tc_ratio, 
          '        Td ratio = %f \n' % args.td_ratio, 
          'number of epochs = %d \n' % args.num_epoch, 
          '              Tc = %d \n' % tc,
          '              Td = %d \n' % td,
          '          Tjoint = %d \n' % t_joint,
          'learn data ratio = %f \n' % args.learned_data_ratio,
          'saving intervals = %d \n' % args.save_interval,
          '-----------------------------------\n',
          ' loaded C model? = %s \n' % (args.c_model_path if args.c_model_path else 'no'),
          ' loaded D model? = %s \n' % (args.d_model_path if args.d_model_path else 'no'),
          '       now epoch = %d \n' % args.current_epoch,
          '-----------------------------------\n',
          '        mailing? = %s \n' % ('disabled' if args.mailing_enabled == False else 'enabled'),
          '====================================',flush=True)

    from train import train, continued_train
    if args.c_model_path and args.d_model_path:
        continued_train(args.dataset_name, args.c_model_path, args.d_model_path, 
                        args.num_epoch, tc, td, args.current_epoch,
                        args.save_interval, 
                        (False if args.mailing_enabled == False else True),
                        args.learned_data_ratio)
    else:
        train(args.dataset_name, 
              args.num_epoch, tc, td,
              args.save_interval, 
              (False if args.mailing_enabled == False else True),
              args.learned_data_ratio)


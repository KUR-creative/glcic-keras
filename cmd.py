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
    help='proportion of local discrimnator. 0 < L < 1',
    type=float, required=True)  

parser.add_argument('-M', '--random_mask_max_proportion',
    help='maximum proportion of random mask size. 0 < M < 1',
    type=float, required=True) 
parser.add_argument('-m', '--random_mask_min_proportion',
    help='minimum proportion of random mask size. 0 < m < 1',
    type=float, required=True)  

parser.add_argument('-a', '--alpha',
    help='joint model hyperparameter(default:0.0004). 0 < a < 1',
    type=float, default=0.0004)  
parser.add_argument('-lr', '--learning_rate',
    help=' learning rate of Dmodel(default:0.01). 0 < lr < 1',
    type=float, default=0.01)  

parser.add_argument('-ne', '--num_epoch',
    help='number of epochs. 0 <= ne',
    type=int, required=True)  
parser.add_argument('-tc', '--tc',
    help='portion of epochs C model(default:0.18). 0 < tc < 1',
    type=float, default=0.18)  
parser.add_argument('-td', '--td',
    help='portion of epochs D model(default:0.02). 0 < td < 1',
    type=float, default=0.02)  
parser.add_argument('-si', '--save_interval',
    help='save interval of epochs. 1 <= si < num_epoch',
    type=int, required=True)  

parser.add_argument('-me', '--mailing_enabled',
    help='is mailing enabled? yes:-me no:(no -me)', 
    type=bool, default=False, nargs='?') 
    # False = (no -me) = disabled, None = -me = enabled.

args = parser.parse_args()

if not (0.0 < args.ld_crop_size_proportion
        and   args.ld_crop_size_proportion < 1.0):
    parser.error('[require] 0.0 < proportion of local discrimnator < 1.0')
if not (0.0 < args.random_mask_max_proportion 
        and   args.random_mask_max_proportion < 1.0):
    parser.error('[require] 0.0 < maximum proportion of mask size < 1.0')
if not (0.0 < args.random_mask_min_proportion 
        and   args.random_mask_min_proportion < 1.0):
    parser.error('[require] 0.0 < minimum proportion of mask size < 1.0')
if not (0.0 < args.alpha and args.alpha < 1.0):
    parser.error('[require] 0.0 < alpha of joint model < 1.0')
if not (0.0 < args.learning_rate): 
    parser.error('[require] 0.0 < learning rate of Discrimnator model')
if not (1 <= args.save_interval and args.save_interval < args.num_epoch): 
    parser.error('[require] 1 <= save interval of epochs. < num_epoch')

if not (0 <= args.num_epoch): 
    parser.error('[require] 0 <= number of epochs')
if not (0.0 < args.tc and args.tc < 1.0):
    parser.error('[require] 0.0 < alpha of joint model < 1.0')
if not (0.0 < args.td and args.td < 1.0):
    parser.error('[require] 0.0 < alpha of joint model < 1.0')

if not (args.random_mask_min_proportion < args.random_mask_max_proportion):
    parser.error('[require] min mask size < max mask size ')
if not (args.tc + args.td < 1.0):
    parser.error('[require] tc + td < 1.0')


if __name__ == "__main__":
    print(' ==============SUMMARY==============\n',
          '    dataset name = %s \n' % args.dataset_name,
          '    dataset type = %s \n' % ('grayscale' if args.is_color_dataset == False else 'rgb'), 
          '      batch size = %d \n' % args.batch_size, 
          '      image size = %d \n' % args.img_size,
          '-----------------------------------\n',
          '    ld crop size = %f \n' % args.ld_crop_size_proportion,
          '  mask max ratio = %f \n' % args.random_mask_max_proportion,
          '  mask min ratio = %f \n' % args.random_mask_min_proportion,
          '-----------------------------------\n',
          'hyperparam alpha = %f \n' % args.alpha, 
          '   hyperparam lr = %f \n' % args.learning_rate,
          '-----------------------------------\n',
          'number of epochs = %d \n' % args.num_epoch, 
          '              Tc = %f \n' % args.tc, 
          '              Td = %f \n' % args.td, 
          'saving intervals = %d \n' % args.save_interval,
          '-----------------------------------\n',
          '        mailing? = %s \n' % ('disabled' if args.mailing_enabled == False else 'enabled'),
          '====================================')


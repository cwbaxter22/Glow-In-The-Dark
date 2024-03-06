from pressure_heatmapping import image_toolbox
from pressure_heatmapping import sv_toolbox
from photo_diode import sv_creation
import numpy as np
import os

#yMinC = 213
#yMaxC = 660
#xMinC = 510
#xMaxC = 860

yMinC = 430
yMaxC = 490
xMinC = 545
xMaxC = 605

#yMinC = 125
#yMaxC = 190
#xMinC = 124
#xMaxC = 175


p_over_int = sv_creation.generate_SV_from_camera_data(yMinC, yMaxC, xMinC, xMaxC)


#photo_diode_data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'photo_diode_data')
#wl_folder = next(os.walk(photo_diode_data_directory))[1][0]
#ox_conc, intensities = sv_creation.create_sv(wl_folder, 500, 750, 650)

raw_data_imgs = image_toolbox.collect_images('heatmap_speed')
raw_amb_imgs = image_toolbox.collect_images('ambient')
raw_dark_imgs = image_toolbox.collect_images('dark_noise')
raw_flat_imgs = image_toolbox.collect_images('flat_field')
raw_WO_imgs = image_toolbox.collect_images('wind_off')

c_data_imgs = image_toolbox.img_crop(raw_data_imgs, yMinC, yMaxC, xMinC, xMaxC)
c_amb_imgs = image_toolbox.img_crop(raw_amb_imgs, yMinC, yMaxC, xMinC, xMaxC)
c_dark_imgs = image_toolbox.img_crop(raw_dark_imgs, yMinC, yMaxC, xMinC, xMaxC)
c_flat_imgs = image_toolbox.img_crop(raw_flat_imgs, yMinC, yMaxC, xMinC, xMaxC)
c_WO_imgs = image_toolbox.img_crop(raw_WO_imgs, yMinC, yMaxC, xMinC, xMaxC)

avg_data_img = image_toolbox.img_avg(c_data_imgs)
avg_amb_img = image_toolbox.img_avg(c_amb_imgs)
avg_dark_img = image_toolbox.img_avg(c_dark_imgs)
avg_flat_img = image_toolbox.img_avg(c_flat_imgs)
avg_WO_img = image_toolbox.img_avg(c_WO_imgs)

corrected_data_image = image_toolbox.flat_field_correction(avg_data_img, avg_amb_img, avg_dark_img, avg_flat_img)
corrected_wind_off_image = image_toolbox.flat_field_correction(avg_WO_img, avg_amb_img, avg_dark_img, avg_flat_img)

#Torr_over_int = sv_creation.convert_to_pressure(ox_conc, intensities, corrected_wind_off_image)

image_toolbox.plot_heatmap(corrected_data_image, corrected_wind_off_image, p_over_int)
#image_toolbox.plot_heatmap(corrected_image, inverse_slope)

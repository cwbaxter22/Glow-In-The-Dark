from pressure_heatmapping import image_toolbox
from pressure_heatmapping import sv_toolbox
from photo_diode import sv_creation
import numpy as np
import os
import matplotlib.pyplot as plt


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

# A is intercept
# B is slope
A, B = sv_creation.generate_SV_from_camera_data(yMinC, yMaxC, xMinC, xMaxC)

# Everything below is PD stuff, ignore for now
#photo_diode_data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'photo_diode_data')
#wl_folder = next(os.walk(photo_diode_data_directory))[1][0]
#ox_conc, intensities = sv_creation.create_sv(wl_folder, 500, 750, 650)
#pressure_torr = np.asarray(ox_conc)*(760/20.9)
#p_p0 = pressure_torr/max(pressure_torr)
#I_I0 = np.asarray(intensities)/np.min(intensities)
#A, B = sv_creation.get_A_B(len(ox_conc), p_p0, I_I0)

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

I0_over_I = corrected_wind_off_image/corrected_data_image
pressure_img = (I0_over_I-A)*(1/B)

#pressure_img = ((1/B)*(corrected_wind_off_image/corrected_data_image)-A)*max(pressure_torr)
plt.imshow(pressure_img, cmap='jet')
plt.colorbar()
plt.show()

#Torr_over_int = sv_creation.convert_to_pressure(ox_conc, intensities, corrected_wind_off_image)

#image_toolbox.plot_heatmap(corrected_data_image, corrected_wind_off_image, p_over_int)

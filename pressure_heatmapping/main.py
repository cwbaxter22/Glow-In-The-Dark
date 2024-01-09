from pressure_heatmapping import image_toolbox
from pressure_heatmapping import sv_toolbox
import numpy as np

yMinC = 70
yMaxC = 440
xMinC = 0
xMaxC = 1180

raw_speed_imgs = image_toolbox.collect_images('heatmap_speed')
raw_amb_imgs = image_toolbox.collect_images('ambient')
raw_dark_imgs = image_toolbox.collect_images('dark_noise')
raw_flat_imgs = image_toolbox.collect_images('flat_field')

c_speed_imgs = image_toolbox.img_crop(raw_speed_imgs, yMinC, yMaxC, xMinC, xMaxC)
c_amb_imgs = image_toolbox.img_crop(raw_amb_imgs, yMinC, yMaxC, xMinC, xMaxC)
c_dark_imgs = image_toolbox.img_crop(raw_dark_imgs, yMinC, yMaxC, xMinC, xMaxC)
c_flat_imgs = image_toolbox.img_crop(raw_flat_imgs, yMinC, yMaxC, xMinC, xMaxC)

avg_speed_img = image_toolbox.img_avg(c_speed_imgs)
avg_amb_img = image_toolbox.img_avg(c_amb_imgs)
avg_dark_img = image_toolbox.img_avg(c_dark_imgs)
avg_flat_img = image_toolbox.img_avg(c_flat_imgs)

corrected_image = image_toolbox.flat_field_correction(avg_speed_img, avg_flat_img, avg_dark_img)

int_to_press_func = sv_toolbox.get_int_function()

heatmap_image = np.vectorize(int_to_press_func)(corrected_image)

image_toolbox.plot_it(heatmap_image)


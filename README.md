Запускать через SRExample.py

В results изображения reference и super для разрешения 512 х 512. 

Оно строится из 6 изображений 256 x 256, пример - low_res_example

Есть проблема, после градиентного спуска фон становится серым. Это проявляется на изображениях с черным фоном и со светлым тоже. 

Сейчас это решается, просто обнулением пикселей, значениях которых близки к 0, но не ноль

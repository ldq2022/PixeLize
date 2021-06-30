# Import Numpy
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import math

# Import Kivy
import kivy

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.config import Config
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty
from kivy.factory import Factory
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.core.window import Window

Config.set('graphics', 'resizable', True)
kivy.require('1.9.1')


class Main(GridLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    file_name = 'None.png'
    style_selected = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2
        self.image_selected = None

        Window.clearcolor = (1, 1, 1, 1)  # window background color

        main_window = BoxLayout(orientation='vertical')
        self.add_widget(main_window)

        # adding empty image to widget
        self.img = Image(source='None.png')
        self.img.allow_stretch = True
        self.img.keep_ratio = True

        self.ori_img = Image(source='None.png')
        self.ori_img.allow_stretch = True
        self.ori_img.keep_ratio = True

        top = BoxLayout(orientation='horizontal')
        top.size_hint = (1, 0.3)
        main_window.add_widget(top)
        top.add_widget(Label())

        mid = BoxLayout(orientation='horizontal')
        mid.size_hint = (1, 2.7)
        main_window.add_widget(mid)
        mid.add_widget(self.ori_img)
        mid.add_widget(self.img)

        bottom = BoxLayout(orientation='horizontal')
        bottom.size_hint = (1, 2)
        main_window.add_widget(bottom)
        bottom_left = BoxLayout(orientation='horizontal')
        bl1 = BoxLayout(orientation='vertical')
        bl2 = BoxLayout(orientation='horizontal')
        bl3 = BoxLayout(orientation='vertical')
        bottom_left.add_widget(bl1)
        bottom_left.add_widget(bl2)
        bottom_left.add_widget(bl3)
        bl1.add_widget(Label())
        bl3.add_widget(Label())
        bl2.size_hint = (5, 1)

        self.dropdown = DropDown()
        sharpen = Button(text='Sharpen', size_hint_y=None, height=70, background_normal='', background_color=(0.7, 0.7, 0.7, 1))
        sharpen.bind(on_release=lambda btn: self.dropdown.select(sharpen.text))
        self.dropdown.add_widget(sharpen)
        blur = Button(text='Blur', size_hint_y=None, height=70, background_normal='', background_color=(0.6, 0.6, 0.6, 1))
        blur.bind(on_release=lambda btn: self.dropdown.select(blur.text))
        self.dropdown.add_widget(blur)
        edge1 = Button(text='Edge Detection 1', size_hint_y=None, height=70, background_normal='', background_color=(0.5, 0.5, 0.5, 1))
        edge1.bind(on_release=lambda btn: self.dropdown.select(edge1.text))
        self.dropdown.add_widget(edge1)
        edge2 = Button(text='Edge Detection 2', size_hint_y=None, height=70, background_normal='', background_color=(0.4, 0.4, 0.4, 1))
        edge2.bind(on_release=lambda btn: self.dropdown.select(edge2.text))
        self.dropdown.add_widget(edge2)
        edge3 = Button(text='Edge Detection 3', size_hint_y=None, height=70, background_normal='', background_color=(0.3, 0.3, 0.3, 1))
        edge3.bind(on_release=lambda btn: self.dropdown.select(edge3.text))
        self.dropdown.add_widget(edge3)

        # create a main button
        self.dropdown_button = Button(text='[b]Select Effects[/b]', markup=True, size_hint=(1, 0.2), pos_hint={'right': 0.975, 'top': 0.6})
        self.dropdown_button.bind(on_release=self.dropdown.open)
        # one last thing, listen for the selection in the dropdown list and
        # assign the data to the button text.

        self.dropdown.bind(on_select=lambda instance, txt: setattr(self.dropdown_button,
                                                                   'text',
                                                                   'Selected Effect: ' + txt))

        bl2.add_widget(self.dropdown_button)

        bottom_mid = BoxLayout(orientation='vertical')
        bottom_mid.size_hint = (0.55, 1)
        bottom_right = BoxLayout(orientation='horizontal')
        bottom_right.size_hint = (0.8, 1)
        bottom.add_widget(bottom_left)
        bottom.add_widget(bottom_mid)
        bottom.add_widget(bottom_right)
        self.slider = Slider(value_track=True, value_track_color=[0.3, 0.8, 1, 1])
        txt = 'Blur Amount:  ' + str(int(self.slider.value)) + '%'
        self.slider_val = Label(text='[b][color=000000]' + txt + '[/color][/b]', markup=True, font_size=29)
        bottom_mid.add_widget(Label())
        bottom_mid.add_widget(self.slider)
        bottom_mid.add_widget(self.slider_val)
        bottom_mid.add_widget(Label())

        br1 = BoxLayout(orientation='vertical')
        br2 = BoxLayout(orientation='vertical')
        br2.size_hint = (1.2, 1)
        br3 = BoxLayout(orientation='vertical')
        bottom_right.add_widget(br1)
        bottom_right.add_widget(br2)
        bottom_right.add_widget(br3)
        br1.add_widget(Label())
        br3.add_widget(Label())

        self.select_file_btn = Button(text="[b]Select File[/b]", markup=True, font_size=29, background_normal='', background_color=(1, 0, 1, 1))
        self.select_file_btn.bind(on_press=self.show_load)
        self.apply_btn = Button(text="[b]Apply[/b]", markup=True, font_size=29, background_normal='', background_color=(1, 0.5, 0, 1))
        self.apply_btn.bind(on_press=self.apply_button)
        br2.add_widget(Label())
        br2.add_widget(self.select_file_btn)
        br2.add_widget(Label())
        br2.add_widget(self.apply_btn)
        br2.add_widget(Label())

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self, instance):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.95, 0.95),
                            pos_hint={'right': 0.975, 'top': 1})
        self._popup.open()

    def load(self, path, filename):
        # with open(os.path.join(path, filename[0])) as stream:
        #     self.text_input.text = stream.read()

        if not filename:
            self.dismiss_popup()
            self.alert()
            return

        self.file_name = filename[0]
        print(self.file_name)

        if self.file_name[-3:] != 'jpg' and self.file_name[-3:] != 'png':
            self.dismiss_popup()
            self.alert()
            return

        self.dismiss_popup()

        self.ori_img.source = self.file_name
        self.ori_img.reload()
        self.img.source = self.file_name
        self.img.reload()



    def save(self, path, filename):
        # with open(os.path.join(path, filename), 'w') as stream:
        #     stream.write(self.text_input.text)

        self.dismiss_popup()

    def alert(self):
        popup = Popup(title='Error',
                      content=Label(text='File format is not PNG or JPG'),
                      size_hint=(0.6, 0.6),
                      pos_hint={'right': 0.8, 'top': 1})
        popup.open()

    def apply_button(self, instance):
        if self.file_name == 'None.png':
            return

        no_change = np.array((
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]), dtype="double")

        blur0 = np.array((
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]), dtype="double")

        sharpen = np.array((
            [0, -2, 0],
            [-2, 9, -2],
            [0, -2, 0]), dtype="int")

        sharpen3 = np.array((
            [0, -3, 0],
            [-3, 13, -3],
            [0, -3, 0]), dtype="int")

        sharpen4 = np.array((
            [0, -4, 0],
            [-4, 17, -4],
            [0, -4, 0]), dtype="int")

        sharpen6 = np.array((
            [0, -6, 0],
            [-6, 25, -6],
            [0, -6, 0]), dtype="int")

        sharpen10 = np.array((
            [0, -10, 0],
            [-10, 41, -10],
            [0, -10, 0]), dtype="int")

        sharpen100 = np.array((
            [0, -100, 0],
            [-100, 401, -100],
            [0, -100, 0]), dtype="int")

        direction1 = np.array((
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]), dtype="float")

        direction3 = np.array((
            [2, 0, -2],
            [2, 0, -2],
            [2, 0, -2]), dtype="float")

        direction2 = np.array((
            [3, 3, 3],
            [0, 0, 0],
            [-3, -3, -3]), dtype="float")

        edge1 = np.array((
            [-2, -2, -2],
            [-2, 16, -2],
            [-2, -2, -2]), dtype="float")

        edge2 = np.array((
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]), dtype="float")

        edge3 = np.array((
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]), dtype="float")

        laplacian = np.array((
            [0, -7, 0],
            [-7, 28, -7],
            [0, -7, 0]), dtype="float")

        rgb_array = cv2.imread(self.file_name)

        if self.slider.value < 5:
            blur0 = np.array((
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]), dtype="double")
            convoleOutput = cv2.filter2D(rgb_array, -1, blur0)
        elif self.slider.value <= 10:
            blur10 = np.ones((2, 2), dtype="float") * (1.0 / (2 * 2))
            convoleOutput = cv2.filter2D(rgb_array, -1, blur10)
        else:
            x = int(self.slider.value)
            v = x // 2
            blurX = np.ones((v, v), dtype="int") * (1.0 / (v * v))
            convoleOutput = cv2.filter2D(rgb_array, -1, blurX)

        modified_img_filename = './new.png'
        if self.dropdown_button.text[17:] == 'Blur':
            cv2.imwrite(modified_img_filename, convoleOutput)
        elif self.dropdown_button.text[17:] == 'Sharpen':
            cv2.imwrite(modified_img_filename, cv2.filter2D(rgb_array, -1, sharpen10))
        elif self.dropdown_button.text[17:] == 'Edge Detection 1':
            cv2.imwrite(modified_img_filename, cv2.filter2D(rgb_array, -1, edge1))
        elif self.dropdown_button.text[17:] == 'Edge Detection 2':
            cv2.imwrite(modified_img_filename, cv2.filter2D(rgb_array, -1, laplacian))
        elif self.dropdown_button.text[17:] == 'Edge Detection 3':
            cv2.imwrite(modified_img_filename, cv2.filter2D(rgb_array, -1, direction2))

        self.ori_img.source = self.file_name
        self.ori_img.reload()
        self.img.source = modified_img_filename
        self.img.reload()

    def update(self):
        self.slider_val.text = '[b][color=000000]' + 'Blur Amount:  ' + str(int(self.slider.value)) + '%' + '[/color][/b]'
        self.slider_val.markup = True

    def on_touch_up(self, touch):
        self.update()

    def convolve(self, image, kernel):
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                   cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        # loop over the input image, "sliding" the kernel across
        # each (x, y)-coordinate from left-to-right and top to
        # bottom
        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                # extract the ROI of the image by extracting the
                # *center* region of the current (x, y)-coordinates
                # dimensions
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                # perform the actual convolution by taking the
                # element-wise multiplicate between the ROI and
                # the kernel, then summing the matrix
                k = (np.multiply(roi, kernel)).sum()
                # store the convolved value in the output (x,y)-
                # coordinate of the output image
                output[y - pad, x - pad] = k

        # rescale the output image to be in the range [0, 255]
        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")
        return output


# creating the App class
class ImageModifier(App):
    def build(self):
        return Main()


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)





Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

# run the app
if __name__ == '__main__':
    ImageModifier().run()

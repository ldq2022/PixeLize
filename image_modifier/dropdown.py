from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.base import runTouchApp



dropdown = DropDown()

sharpen = Button(text='Sharpening', size_hint_y=None, height=70)
sharpen.bind(on_release=lambda btn: dropdown.select(sharpen.text))
dropdown.add_widget(sharpen)

blur = Button(text='Blurring', size_hint_y=None, height=70)
blur.bind(on_release=lambda btn: dropdown.select(blur.text))
dropdown.add_widget(blur)

# create a big main button
dropdown_button = Button(text='Select ', size_hint_y=None)

# show the dropdown menu when the main button is released
# note: all the bind() calls pass the instance of the caller (here, the
# mainbutton instance) as the first argument of the callback (here,
# dropdown.open.).
dropdown_button.bind(on_release=dropdown.open)

# one last thing, listen for the selection in the dropdown list and
# assign the data to the button text.
dropdown.bind(on_select=lambda instance, txt: setattr(dropdown_button, 'text', txt))

runTouchApp(dropdown_button)

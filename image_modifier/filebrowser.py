from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup

import os

file_name = None

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)


    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.95, 0.95),
                            pos_hint={'right': 0.975, 'top': 1})
        self._popup.open()



    def load(self, path, filename):
        # with open(os.path.join(path, filename[0])) as stream:
        #     self.text_input.text = stream.read()
        global file_name

        if not filename:
            self.dismiss_popup()
            self.alert()
            return


        file_name = filename[0]
        print(file_name)


        if file_name[-3:] != 'jpg' and file_name[-3:] != 'png':
            self.dismiss_popup()
            self.alert()
            return

        self.dismiss_popup()
        App.get_running_app().stop()

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

class Editor(App):
    pass


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)


if __name__ == '__main__':
    Editor().run()

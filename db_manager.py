import sqlite3
from colorama import Fore
# from PIL import Image
# import io

from numpy import imag

class database_manager():
    def __init__(self):
        self.connection = sqlite3.connect('image_data.db')
        self.cursor = self.connection.cursor()
        
    def create_database(self):
        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS image_table(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    name TEXT,
                    image BLOP UNIQUE) """)
        self.close_db()
        
    def path_to_image(self, path):
        with open(path,'rb') as f:
            data = f.read()
        return data
        
    def add_image(self, name, path=None,img=None):
        if path is not None:
            data = self.path_to_image(path)
        if img is not None:
            data= img
        try:
            self.cursor.execute("""INSERT INTO image_table (name, image) VALUES (?,?)""",(name, data))
        except sqlite3.IntegrityError:
            print(f'{Fore.RED}ERROR:{Fore.WHITE} Image Already Exist in Database')
        self.close_db()
    
    # def name_image_from_db(self):
    #     data=self.cursor.execute("""SELECT * FROM image_table """)
    #     name_list=[]
    #     image_list=[]
    #     for id, name, image in data:
    #         name_list.append(name)
    #         # print(image[:10])
    #         img= Image.open(io.BytesIO(image))
    #         img1 = list(img.getdata())
    #         image_list.append(img1)
    #     self.close_db()
    #     return name_list, image_list
    
    def delete_on_name(self, name):
        self.cursor.execute("""DELETE FROM image_table WHERE name = ?""",(name,))
        self.close_db()
        
    def delete_on_id(self, id):
        self.cursor.execute("""DELETE FROM image_table WHERE id = ?""",(id,))
        self.close_db()
        
    def delete_on_image(self, path):
        data = self.path_to_image(path)
        self.cursor.execute("""DELETE FROM image_table WHERE image = ?""",(data,))
        self.close_db()
            
    def close_db(self):
        self.connection.commit()
        self.cursor.close()
        self.connection.close()
    
    def write(self, path):
        data=self.cursor.execute("""SELECT * FROM image_table """)
        for _, name, image in data:
            with open(f'{path}/{name}.jpg','wb') as f:
                f.write(image)
    
        
                
        

        
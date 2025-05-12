from dataclay import Client
from model.company import Employee, Company

# Conectar con el servidor de DataClay
client = Client(host="localhost", port=6867, username="testuser", password="s3cret", dataset="testdata")

# Mostrar si la conexión fue exitosa
client.start()
print("Conexión exitosa con el servidor de DataClay")
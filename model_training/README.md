✅ 1. Crear un entorno virtual en tu proyecto
Desde la terminal, estando en la raíz de tu proyecto (model_training):

bash
Copiar
Editar
python3 -m venv env
Esto crea una carpeta env/ con un entorno virtual aislado.

✅ 2. Activar el entorno virtual
En macOS / zsh:
bash
Copiar
Editar
source env/bin/activate
Verás que ahora el prompt te muestra algo como:

bash
Copiar
Editar
(env) carlosqp@Mac-Mini-M4 model_training %
✅ 3. Crear requirements.txt
Crea un archivo llamado requirements.txt con este contenido mínimo para tu proyecto:


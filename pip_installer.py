import subprocess

# List of required libraries
libraries = ['tensorflow', 'keras', 'mplfinance', 'tkinter', 'pandas', 'numpy']

# Install libraries
for library in libraries:
    try:
        subprocess.check_call(['pip', 'install', library])
        print(f'Successfully installed {library}')
    except subprocess.CalledProcessError:
        print(f'Error installing {library}')

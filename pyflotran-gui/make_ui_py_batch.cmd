for %%f in (*.ui) do (

        echo %%~nf
        pyside-uic %%~nf.ui > %%~nf.py
)

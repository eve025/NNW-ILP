Notas para actualizar Protobuf gencode
====================================

Síntomas:
- Mensajes como: "Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1"

Pasos rápidos y no invasivos para reducir las advertencias:

1) Actualizar el paquete Python `protobuf` (a menudo corrige estas advertencias):

```powershell
py -3 -m pip install --upgrade pip
py -3 -m pip install --upgrade protobuf grpcio-tools
```

2) Si el proyecto contiene archivos generados `_pb2.py` que fueron creados con un `protoc` antiguo (gencode), regenéralos usando una versión reciente de `protoc` (o `grpc_tools.protoc`). Ejemplo:

```powershell
py -3 -m pip install --upgrade grpcio-tools
py -3 -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/tuarchivo.proto
```

3) Si no puedes regenerar las fuentes de inmediato, la advertencia no es crítica por ahora; programa la regeneración de los protos y fija versiones compatibles de `protobuf` en `requirements.txt` para producción.

Notas:
- Siempre prueba después de actualizar protobuf: la API o el comportamiento puede cambiar entre versiones mayores.
- En Windows, asegúrate de usar el binario `protoc` o `grpc_tools` para generar fuentes Python que coincidan con la versión runtime instalada.

def make_dist():
    return default_python_distribution()

def make_exe(dist):
    policy = dist.make_python_packaging_policy()
    policy.resources_location = "in-memory"
    policy.resources_location_fallback = "filesystem-relative:lib"

    python_config = dist.make_python_interpreter_config()
    python_config.run_module = "app.main"

    exe = dist.to_python_executable(
        name="faiss-proxy",
        packaging_policy=policy,
        config=python_config,
    )

    # Include Python module dependencies
    exe.add_python_resources(exe.pip_install([
        "fastapi",
        "uvicorn",
        "faiss-cpu",
        "pydantic",
        "python-multipart",
        "python-jose[cryptography]",
        "numpy",
        "psutil",
    ]))

    # Include the application code
    exe.add_python_resources(exe.pip_install(["."], []))

    return exe

def make_embedded_resources(exe):
    return exe.to_embedded_resources()

def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files

def register_code_signers():
    return []

register_target("exe", make_exe)
register_target("resources", make_embedded_resources, depends=["exe"], default=True)
register_target("install", make_install, depends=["exe"])
register_target("dist", make_dist)

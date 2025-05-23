import os


def clean(project_path):

    directory = [".github", "imgs", ".faq", "assets"]
    files = [".coveragerc", ".gitignore", ".readthedocs.yml", ".pre-commit-config.yaml", "Makefile", "make.bat", "tox.ini", "asv.conf.json", "asvhashfile", "faq.yml", "LICENSE"]
    entries = list(os.listdir(project_path))
    for entry in entries:
        try:
            entry_path = os.path.join(project_path, entry)
            if os.path.isdir(entry_path) and entry in directory:
                try:
                    for root, dir_name, file_list in os.walk(entry_path, topdown=False):
                        for f in file_list:
                            try:
                                file = os.path.join(root, f)
                                os.remove(file)
                            except Exception as e:
                                print(f"Failed to remove:{f}: {e}")
                        for d in dir_name:
                            try:
                                dir_name = os.path.join(root, d)
                                os.rmdir(dir_name)
                            except Exception as e:
                                print(f"Failed to remove: {d}: {e}")
                    os.rmdir(entry_path)
                except Exception as e:
                    print(f"Error processing: {entry}: {e}")
            elif os.path.isfile(entry_path):
                if entry in files:
                    try:
                        os.remove(entry_path)
                    except Exception as e:
                        print(f"Failed to remove: {entry}: {e}")
        except Exception as e:
            print(f"Error processing:  {entry}: {e}")


#
# if __name__ == "__main__":
#     path = "projects/typer"
#     clean(path)
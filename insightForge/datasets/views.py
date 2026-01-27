import pandas as pd
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage

def upload_dataset(request):
    context = {}

    if request.method == "POST" and request.FILES.get("dataset"):
        file = request.FILES["dataset"]
        file_path = default_storage.save(file.name, file)

        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
            else:
                context["error"] = "Unsupported file format"
                return render(request, "datasets/upload.html", context)

            # Clear old session data when uploading a new dataset
            old_keys = list(request.session.keys())
            for key in old_keys:
                if key != '_auth_user_id' and key != '_auth_user_backend' and key != '_auth_user_hash':
                    del request.session[key]

            # Save dataframe to session
            request.session["dataset"] = df.to_json()
            context["preview"] = df.to_html(classes="table table-bordered")
            context["rows"] = len(df)
            context["cols"] = len(df.columns)
            context["upload_success"] = True

        except Exception as e:
            context["error"] = str(e)

    return render(request, "datasets/upload.html", context)

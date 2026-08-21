
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User

from .auth_mongo import authenticate_user, create_user


def login_view(request):
    # =========================================================================
    # CRITICAL: Clear any leftover messages from previous pages
    # =========================================================================
    # This prevents messages from outlier detection or other pages
    # from appearing on the login page
    storage = messages.get_messages(request)
    for _ in storage:
        pass  # Consume all messages without using them
    
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]

        # 1️⃣ Try MongoDB
        try:
            mongo_user = authenticate_user(email, password)
            if mongo_user:
                request.session["mongo_user"] = mongo_user["email"]
                return redirect("upload_dataset")
        except Exception:
            pass  # MongoDB down → fallback

        # 2️⃣ Django fallback
        user = authenticate(
            request,
            username=email,
            password=password
        )
        if user:
            login(request, user)
            return redirect("upload_dataset")

        messages.error(request, "Invalid credentials")

    return render(request, "accounts/login.html")


def register_view(request):
    # =========================================================================
    # CRITICAL: Clear any leftover messages from previous pages
    # =========================================================================
    # This prevents messages from outlier detection or other pages
    # from appearing on the registration page
    storage = messages.get_messages(request)
    for _ in storage:
        pass  # Consume all messages without using them
    
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]

        # 1️⃣ Try MongoDB
        try:
            success, msg = create_user(username, email, password)
            if success:
                messages.success(request, "Registered successfully")
                return redirect("login")
        except Exception:
            pass

        # 2️⃣ Django fallback
        if User.objects.filter(username=email).exists():
            messages.error(request, "User already exists")
        else:
            User.objects.create_user(
                username=email,
                email=email,
                password=password
            )
            messages.success(request, "Registered successfully")
            return redirect("login")

    return render(request, "accounts/register.html")

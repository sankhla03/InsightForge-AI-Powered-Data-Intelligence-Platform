def require_login(request):
    """
    Allows Django auth OR MongoDB session auth
    """
    if request.user.is_authenticated:
        return True

    if "mongo_user" in request.session:
        return True

    return False
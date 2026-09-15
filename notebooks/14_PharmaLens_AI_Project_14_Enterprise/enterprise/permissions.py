ROLE_PERMISSIONS = {
    "admin": {"dashboard","market","brand","molecule","company","forecast","launch","gtm",
              "recommendation","similarity","agent","reports","upload","users","subscription"},
    "manager": {"dashboard","market","brand","molecule","company","forecast","launch","gtm",
                "recommendation","similarity","agent","reports","upload"},
    "analyst": {"dashboard","market","brand","molecule","forecast","launch","gtm",
                "recommendation","similarity","agent","reports"},
    "viewer": {"dashboard","market","brand","molecule","company","forecast","launch","gtm",
               "recommendation","similarity","reports"},
}

def has_permission(user, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(user.role, set())

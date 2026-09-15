PLAN_LIMITS = {
    "free": {"max_users": 1, "max_datasets": 1, "max_reports": 3},
    "professional": {"max_users": 10, "max_datasets": 25, "max_reports": 100},
    "enterprise": {"max_users": 100, "max_datasets": 1000, "max_reports": 10000},
}

def get_plan_limits(plan: str):
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

def check_limit(current_count: int, plan: str, resource: str) -> bool:
    maximum = get_plan_limits(plan).get(resource)
    return True if maximum is None else current_count < maximum

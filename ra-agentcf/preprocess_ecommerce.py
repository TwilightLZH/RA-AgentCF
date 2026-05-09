import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def clean_text(value, default="unknown"):
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def join_sequence(items):
    return " ".join(str(item) for item in items)


def quantile_or_zero(series, q):
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def humanize_brand(value):
    brand = clean_text(value)
    if brand == "unknown":
        return ""
    upper_brands = {"lg", "bq", "hp", "htc", "jbl", "oppo", "tp-link"}
    if brand.lower() in upper_brands:
        return brand.upper()
    return brand.title()


def humanize_category(value):
    category = clean_text(value, "unknown category")
    if category == "unknown category":
        return "general product"

    parts = [part.replace("_", " ").strip() for part in category.split(".") if part.strip()]
    broad_terms = {
        "accessories",
        "appliances",
        "apparel",
        "auto",
        "computers",
        "construction",
        "electronics",
        "furniture",
        "kids",
        "sport",
    }
    if len(parts) > 1 and parts[0] in broad_terms:
        parts = parts[1:]

    label = " ".join(parts)
    phrase_replacements = {
        "kitchen refrigerators": "kitchen refrigerator",
        "kitchen washer": "kitchen washer",
        "kitchen toster": "kitchen toaster",
        "tools light": "tool light",
        "shoes slipons": "slip-on shoes",
        "living room cabinet": "living room cabinet",
        "environment air conditioner": "air conditioner",
        "audio headphone": "headphones",
        "components power supply": "computer power supply",
        "peripherals printer": "printer",
        "fmcg diapers": "diapers",
        "bedroom blanket": "bedroom blanket",
    }
    label = phrase_replacements.get(label, label)
    return label


def build_product_title(brand, category, product_id):
    brand_text = humanize_brand(brand)
    category_text = humanize_category(category)
    if brand_text:
        return f"{brand_text} {category_text} #{product_id}"
    return f"{category_text.title()} #{product_id}"


def build_examples(events, max_history_items=None):
    """Convert each user's event stream into sequential recommendation samples.

    Only purchase events become supervised targets. Both view and purchase
    events remain in the history sequence so views provide weak-interest
    context while purchases provide strong positive feedback.
    """
    examples_by_user = defaultdict(list)
    for user_id, group in events.groupby("user_id", sort=False):
        history = []
        for row in group.itertuples(index=False):
            product_id = str(row.product_id)
            # RecBole masks historical items during full-sort evaluation. If
            # the target product stays in history, the ground truth can be
            # masked by mistake, so remove the current target from its history.
            history_for_target = [item for item in history if item != product_id]
            if max_history_items is not None:
                history_for_target = history_for_target[-max_history_items:]
            if row.event_type == "purchase" and history_for_target:
                examples_by_user[str(user_id)].append(
                    {
                        "user_id": str(user_id),
                        "item_id_list": history_for_target,
                        "item_id": product_id,
                        "event_time": row.event_time,
                    }
                )
            # Consecutive duplicate views usually represent dwell/refresh
            # behavior. Compress them to keep a cleaner interest sequence.
            if not history or history[-1] != product_id:
                history.append(product_id)
                if max_history_items is not None:
                    history = history[-max_history_items:]
    return examples_by_user


def split_examples(
    examples_by_user,
    min_purchases,
    max_users=None,
    max_train_examples_per_user=None,
):
    train_rows, valid_rows, test_rows = [], [], []
    eligible_users = [
        (user_id, examples)
        for user_id, examples in examples_by_user.items()
        if len(examples) >= min_purchases
    ]
    # Prefer users with more purchase examples to stabilize small experiments.
    eligible_users.sort(key=lambda item: (-len(item[1]), int(item[0])))
    if max_users is not None:
        eligible_users = eligible_users[:max_users]

    for user_id, examples in eligible_users:
        user_train_rows = examples[:-2]
        if max_train_examples_per_user is not None:
            # Keep newer training purchases so train/valid/test are closer in
            # time for each selected user.
            user_train_rows = user_train_rows[-max_train_examples_per_user:]
        train_rows.extend(user_train_rows)
        valid_rows.append(examples[-2])
        test_rows.append(examples[-1])
    return train_rows, valid_rows, test_rows


def write_inter(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for row in rows:
            file.write(
                f"{row['user_id']}\t{join_sequence(row['item_id_list'])}\t{row['item_id']}\n"
            )


def item_universe(rows):
    items = set()
    for row in rows:
        items.add(str(row["item_id"]))
        items.update(str(item) for item in row["item_id_list"])
    return items


def build_item_stats(events):
    purchase_events = events[events["event_type"] == "purchase"]
    stats = {}
    for product_id, group in events.groupby("product_id"):
        product_id = str(product_id)
        purchases = purchase_events[purchase_events["product_id"].astype(str) == product_id]
        positive_prices = purchases.loc[purchases["price"] > 0, "price"]
        if positive_prices.empty:
            positive_prices = group.loc[group["price"] > 0, "price"]
        revenue = float(positive_prices.median()) if not positive_prices.empty else 0.0
        view_count = int((group["event_type"] == "view").sum())
        purchase_count = int((group["event_type"] == "purchase").sum())
        sample = group.iloc[0]
        brand = clean_text(sample["brand"])
        category = clean_text(sample["category_code"], "unknown category")
        title = build_product_title(brand, category, product_id)
        stats[product_id] = {
            "title": title,
            "category": category,
            "brand": brand,
            "price": revenue,
            "view_count": view_count,
            "purchase_count": purchase_count,
        }
    return stats


def write_item_file(path, item_stats, items):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "item_id:token",
                "title:token_seq",
                "category:token_seq",
                "brand:token",
                "price:float",
                "view_count:float",
                "purchase_count:float",
            ]
        )
        for item_id in sorted(items, key=lambda value: int(value)):
            stat = item_stats[item_id]
            writer.writerow(
                [
                    item_id,
                    stat["title"],
                    stat["category"],
                    stat["brand"],
                    f"{stat['price']:.4f}",
                    stat["view_count"],
                    stat["purchase_count"],
                ]
            )


def write_revenue_files(output_dir, dataset_name, item_stats, items):
    revenue_path = output_dir / f"{dataset_name}.revenue.csv"
    behavior_path = output_dir / f"{dataset_name}.item_behavior.csv"
    with revenue_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "revenue"])
        for item_id in sorted(items, key=lambda value: int(value)):
            writer.writerow([item_id, f"{item_stats[item_id]['price']:.4f}"])
    with behavior_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "view_count", "purchase_count"])
        for item_id in sorted(items, key=lambda value: int(value)):
            stat = item_stats[item_id]
            writer.writerow(
                [
                    item_id,
                    stat["view_count"],
                    stat["purchase_count"],
                ]
            )
    return revenue_path, behavior_path


def write_user_profiles(path, events, active_users):
    purchase_events = events[events["event_type"] == "purchase"]
    rows = []
    for user_id in sorted(active_users, key=lambda value: int(value)):
        user_events = events[events["user_id"].astype(str) == user_id]
        user_purchases = purchase_events[purchase_events["user_id"].astype(str) == user_id]
        prices = user_purchases.loc[user_purchases["price"] > 0, "price"]
        view_count = int((user_events["event_type"] == "view").sum())
        purchase_count = int((user_events["event_type"] == "purchase").sum())
        rows.append(
            {
                "user_id": user_id,
                "median_purchase_price": quantile_or_zero(prices, 0.50),
                "q75_purchase_price": quantile_or_zero(prices, 0.75),
                "mean_purchase_price": float(prices.mean()) if not prices.empty else 0.0,
                "view_count": view_count,
                "purchase_count": purchase_count,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_random_candidates(path, rows, all_items, candidate_size, seed):
    random.seed(seed)
    user_positive = defaultdict(set)
    users = set()
    for row in rows:
        users.add(row["user_id"])
        user_positive[row["user_id"]].add(row["item_id"])

    with path.open("w", encoding="utf-8") as file:
        for user_id in sorted(users, key=lambda value: int(value)):
            candidates = [item for item in all_items if item not in user_positive[user_id]]
            random.shuffle(candidates)
            file.write(f"{user_id}\t{join_sequence(candidates[:candidate_size])}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ra-agentcf/dataset/2020-Apr-final.csv")
    parser.add_argument("--dataset", default="ECommerceRA")
    parser.add_argument("--min-purchases", type=int, default=3)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-train-examples-per-user", type=int, default=None)
    parser.add_argument("--max-history-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(input_path)
    events = events.drop_duplicates().copy()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")
    events = events.dropna(subset=["event_time", "user_id", "product_id", "event_type"])
    events = events[events["event_type"].isin(["view", "purchase"])].copy()
    events["price"] = pd.to_numeric(events["price"], errors="coerce").fillna(0.0)
    events = events.sort_values(["user_id", "event_time", "product_id"]).reset_index(drop=True)

    examples_by_user = build_examples(events, args.max_history_items)
    train_rows, valid_rows, test_rows = split_examples(
        examples_by_user,
        args.min_purchases,
        args.max_users,
        args.max_train_examples_per_user,
    )
    all_rows = train_rows + valid_rows + test_rows
    active_users = {row["user_id"] for row in all_rows}
    items = item_universe(all_rows)
    item_stats = build_item_stats(events)

    write_inter(output_dir / f"{args.dataset}.train.inter", train_rows)
    write_inter(output_dir / f"{args.dataset}.valid.inter", valid_rows)
    write_inter(output_dir / f"{args.dataset}.test.inter", test_rows)
    write_item_file(output_dir / f"{args.dataset}.item", item_stats, items)
    revenue_path, behavior_path = write_revenue_files(output_dir, args.dataset, item_stats, items)
    user_profile_path = output_dir / f"{args.dataset}.user_profile.csv"
    write_user_profiles(user_profile_path, events, active_users)
    write_random_candidates(
        output_dir / f"{args.dataset}.random",
        valid_rows + test_rows,
        sorted(items, key=lambda value: int(value)),
        args.candidate_size,
        args.seed,
    )

    purchase_counts = Counter(row["user_id"] for row in all_rows)
    print(f"Generated dataset: {args.dataset}")
    print(f"Output dir: {output_dir}")
    print(f"Users: {len(active_users)}")
    print(f"Items in generated interactions: {len(items)}")
    print(f"Train/valid/test rows: {len(train_rows)}/{len(valid_rows)}/{len(test_rows)}")
    print(f"Min generated purchase examples per user: {min(purchase_counts.values())}")
    print(f"Revenue file: {revenue_path}")
    print(f"Item behavior file: {behavior_path}")
    print(f"User profile file: {user_profile_path}")


if __name__ == "__main__":
    main()

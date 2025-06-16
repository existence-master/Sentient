# sample_data.py
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
import json

def get_quickstart_episodes():
    """Returns the episodes from the Quickstart guide."""
    return [
        {
            'name': 'Freakonomics Radio 1',
            'content': 'Kamala Harris is the Attorney General of California. She was previously '
                       'the district attorney for San Francisco.',
            'type': EpisodeType.text,
            'description': 'podcast transcript',
        },
        {
            'name': 'Freakonomics Radio 2',
            'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
            'type': EpisodeType.text,
            'description': 'podcast transcript',
        },
        {
            'name': 'Freakonomics Radio 3',
            'content': {
                'name': 'Gavin Newsom',
                'position': 'Governor',
                'state': 'California',
                'previous_role': 'Lieutenant Governor',
                'previous_location': 'San Francisco',
            },
            'type': EpisodeType.json,
            'description': 'podcast metadata',
        },
        {
            'name': 'Freakonomics Radio 4',
            'content': {
                'name': 'Gavin Newsom',
                'position': 'Governor',
                'term_start': 'January 7, 2019',
                'term_end': 'Present',
            },
            'type': EpisodeType.json,
            'description': 'podcast metadata',
        },
    ]

def get_bulk_episodes_from_product_data():
    """Returns RawEpisode objects for bulk loading from sample product data."""
    product_data = [
        {
            "id": "PROD001",
            "name": "Men's SuperLight Wool Runners",
            "color": "Dark Grey",
            "material": "Wool",
            "price": 125.00,
            "in_stock": True,
        },
        {
            "id": "PROD002",
            "name": "Women's Tree Breezers",
            "color": "Rugged Beige",
            "material": "TENCEL™ Lyocell",
            "price": 100.00,
            "in_stock": True,
        }
    ]
    return [
        RawEpisode(
            name=f"Product Update - {product['id']}",
            content=json.dumps(product),
            source=EpisodeType.json,
            source_description="Allbirds product catalog update",
            reference_time=datetime.now(timezone.utc),
        ) for product in product_data
    ]
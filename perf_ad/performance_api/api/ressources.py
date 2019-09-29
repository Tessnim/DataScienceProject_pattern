# api/resources.py
from tastypie.resources import ModelResource
from .models import Performance

from tastypie.constants import ALL
from tastypie.serializers import Serializer


class PerformanceResource(ModelResource):
    class Meta:
        queryset = Performance.objects.all()
        resource_name = 'performance'
        # Performance.objects.filter(date__range=[start_date, end_date]) this is in django native
        filtering = {
            "date": ['range'],
            "channel": ALL,
            "country": ALL,
            "os": ALL
        }
        # ordering by any column
        ordering = ['date', 'channel', 'country', 'os', 'impressions', 'clicks', 'installs', 'spend', 'revenue']
        serializer = Serializer(formats=['json'])

    def hydrate(self, bundle):
        print(bundle.request)
        return bundle

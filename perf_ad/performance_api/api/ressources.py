# api/resources.py
from tastypie.resources import ModelResource
from .models import Performance
from datetime import datetime
from tastypie.constants import ALL, ALL_WITH_RELATIONS
from datetime import datetime


class PerformanceResource(ModelResource):
    class Meta:
        # queryset = Performance.objects.filter(date__range=().groupby().orderby())
        queryset = Performance.objects.all()
        resource_name = 'performance'
        filtering = {
            "date": 'range',
            "title": ALL,
            "channels": ALL,
            "countries": ALL,
            "os": ALL
        }

    def get_object_list(self, request):
        if 'date_from' in request.POST and 'date_to' in request.POST:  # filter by date range
            dfrom = request.POST['date_from']
            dto = request.POST['date_to']
            date_from = datetime.strptime(dfrom, "YYYY-MM-DD")
            date_to = datetime.strptime(dto, "YYYY-MM-DD")
            return super(PerformanceResource, self).get_object_list(request).filter(start_date__gte=date_from,
                                                                                    start_date__lte=date_to)
        # if 'title' in request.POST:
        #     title = request.POST['title']
        # if 'channel' in request.POST:  # channel, country, os
        #     channel = request.POST['channel']
        # if 'country' in request.POST:
        #     country = request.POST['country']
        # if 'os' in request.POST:
        #     os = request.POST['os']
        # return super(PerformanceResource, self).get_object_list(request).filter(start_date__gte=date_from,
        #                                                                         start_date__lte=date_to)

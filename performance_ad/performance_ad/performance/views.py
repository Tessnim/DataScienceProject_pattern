from django.shortcuts import render, HttpResponse
import csv, io
from django.contrib import messages
from django.contrib.auth.decorators import permission_required

from .models import Performance

#(date_string, format)


def filter_by_date_range(request, date_from, date_to):
    # filter by time range(date_from / date_to is enough), channels, countries, operating systems
    # date format should be: YYYY-MM-DD
    if request.method == "GET":
        # date_from = strptime(date_from, "YYYY-MM-DD")
        # date_to = strptime(date_to, "YYYY-MM-DD")
        query_set = Performance.objects.filter(date__range=[date_from, date_to])
        context = {"objects": query_set}

    return render(request, 'performance/filter_by_date.html', context)


def performance_details(request):
    if request.method == "GET":
        print("return all dataset")
    objects = Performance.objects.all()
    context = {"objects": objects}
    return render(request, 'performance/list.html', context)


@permission_required('admin.can_add_log_entry')
def data_upload(request):
    template = 'performance/data_upload.html'

    prompt = {
        'order': 'csv from technical test'
    }

    if request.method == "GET":
        return render(request, template, prompt)

    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, "This is not a csv file")

    dataset = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(dataset)

    next(io_string) # skip the first line of csv file (header)

    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        _, created = Performance.objects.update_or_create(
            date=column[0],
            channel=column[1],
            country=column[2],
            os=column[3],
            impressions=column[4],
            clicks=column[5],
            installs=column[6],
            spend=column[7],
            revenue=column[8],
        )

    context = {}
    print("done")
    return render(request, template, context)


---
title : "Python : datetime"
categories :
- DAILY
tag : [basic, coding]
toc: true
toc_sticky: true
toc_label : "목록"
author_profile : false
search: true
use_math: true
---
<br/>

# Python : datetime


## 1. 내용
맨날 구글링하기 귀찮으니까 작성해놓습니다.
- datetime 모듈에는 date, timedelta, tzinfo, timezone 객체가 있음.
- 쓰잘데 없는 것 제외하고 datetime.datetime(), datetime.timedelta(), .strftime(), strptime()만 알고 있으면 됨.

## 2. date 객체 
- date 객체의 year, month, day는 모두 필수 인자이고, 인자는 해당 날짜가 가질수 있는 범위 내의 정수여야함.
- date.today()로 오늘 날짜 출력함.
- .replace(필수 인자=25)로 날짜 변환함.
- .weekday()로 요일을 숫자로 출력함. (월요일은 0 일요일은 6)
- .ctime()으로 날짜를 문자로 출력함. 
```python 
import datetinme

datetime.date(2002, 12, 31)
# datetime.date(2002, 12, 31)

datetime.date.today()
# datetime.date(2023, 11, 2)

datetime.date(2002, 12, 31).replace(day=26)
# datetime.date(2002, 12, 26)

datetime.date(2023, 11, 2).weekday()
# 3. -> 월요일은 0 일요일은 6

datetime.date(2023, 11, 2).ctime()
# 'Thu Nov  2 00:00:00 2023'
```

## 3. datetime 객체 
- date 객체의 year, month, day는 모두 필수 인자이고, 그 외에도 hour, minute, second, microsecond, tzinfo, fold가 더 있음.
- 대부분 date 객체와 비슷함.
- .today()와 .now()는 거의 동일함.
- .combine()은 date객체와 time객체를 결합함.
- .strptime()은 다양한 포맷의 문자로 된 날짜를 날짜 형식으로 변환함. 
  - 주로 문자로 된 날짜 열을 데이터 프레임의 인덱스로 넣기 위해 변환할 때 사용할 것 같음.
- .strftime()은 주어진 포맷에 따라 객체를 문자열로 변환함.
  - 말 그대로 내가 원하는 문자열로 변환할 때 사용할듯.
- 
```python 
import datetinme

datetime.datetime(2023, 11, 2)
# datetime.datetime(2023, 11, 2, 0, 0)

datetime.datetime.today()
datetime.datetime.now()
# datetime.datetime(2023, 11, 2, 11, 19, 11, 397273)

datetime.datetime.combine(datetime.date(2023, 11, 2), datetime.time(12, 00, 00))
# datetime.datetime(2023, 11, 2, 12, 0)

datetime.datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M")
# datetime.datetime(2006, 11, 21, 16, 30)

datetime.datetime.now().strftime("%d/%m/%y %H:%M")
# '02/11/23 11:35'

```

## 4. datetime.timedelta 객체 
- 날짜와 시간을 연산하는 메서드임.
- .total_seconds()는 기간 동안의 총 시간을 초로 반환함. datetime.timedelta(seconds=1)로 나눈 것과 같음.
  
```python 
import datetinme

datetime.date.today() + datetime.timedelta(days=1)
# datetime.date(2023, 11, 3)

datetime.datetime.now() + datetime.timedelta(days=1)
# datetime.datetime(2023, 11, 3, 13, 52, 53, 471782)

datetime.timedelta(days=1).total_seconds()
# 86400.0

datetime.timedelta(days=1)/datetime.timedelta(seconds=1)
# 86400.0

```
## 참고
- 이제 구글링 안해도 될 듯!
- 

  
---
title : "[SQL]-(084) SELECT FOR UPDATE절 이해하기"
excerpt: "SELECT FOR UPDATE절 이해하기"
categories :
- SQL
tag : [sql, basic]
toc: true
toc_sticky: true
toc_label : "목록"
author_profile : false
search: true
---

---
**[Reference]** 초보자를 위한 SQL 200제

---

### [SQL 문법]
검색하는 행에 락을 걸 수 있다.  
SELECT FOR UPDATE문은 WHERE절로 검색하는 행에 락을 거는 SQL 문이다.  
해당 문이 실행되는 동안 다른 터미널 창에서 같은 조건의 행을 UPDATE할 수 없다.
```sql
SELECT ename, sal, deptno
  FROM emp
  WHERE ename='JONES'
  FOR UPDATE
```
### [예시]
```python
```
### [결과]

    

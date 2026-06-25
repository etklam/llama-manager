from ui_helpers import log_bus


def test_emit_no_subscribers_is_noop():
    log_bus._subscribers.clear()
    log_bus.emit("INFO", "hello")


def test_emit_calls_every_subscriber_once():
    log_bus._subscribers.clear()
    calls = []
    log_bus.subscribe(lambda level, msg: calls.append((level, msg)))
    log_bus.emit("INFO", "a")
    log_bus.emit("ERROR", "b")
    assert calls == [("INFO", "a"), ("ERROR", "b")]


def test_subscribe_returns_unsubscribe_fn():
    log_bus._subscribers.clear()
    calls = []
    unsub = log_bus.subscribe(lambda level, msg: calls.append((level, msg)))
    assert callable(unsub)
    log_bus.emit("INFO", "x")
    unsub()
    log_bus.emit("INFO", "y")
    assert calls == [("INFO", "x")]


def test_unsubscribe_is_idempotent():
    log_bus._subscribers.clear()
    calls = []
    fn = lambda level, msg: calls.append(msg)
    unsub = log_bus.subscribe(fn)
    unsub()
    unsub()
    log_bus.emit("INFO", "z")
    assert calls == []


def test_emit_fanout_in_subscription_order():
    log_bus._subscribers.clear()
    order = []
    log_bus.subscribe(lambda l, m: order.append("first"))
    log_bus.subscribe(lambda l, m: order.append("second"))
    log_bus.subscribe(lambda l, m: order.append("third"))
    log_bus.emit("INFO", "go")
    assert order == ["first", "second", "third"]


def test_multiple_subscribers_each_called():
    log_bus._subscribers.clear()
    a = []
    b = []
    log_bus.subscribe(lambda l, m: a.append(m))
    log_bus.subscribe(lambda l, m: b.append(m))
    log_bus.emit("INFO", "fan")
    assert a == ["fan"]
    assert b == ["fan"]
